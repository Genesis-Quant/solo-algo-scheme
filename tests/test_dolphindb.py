"""可选真实插件验证：测试会话内造确定性行情，不写生产数据库。"""

import json
import os
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel
from runtime.apps.query import FactorQuery
from runtime.database import CORE_TABLE

from scheme import (
    Algo,
    Direction,
    DosVar,
    Factor,
    FactorParams,
    LimitOrder,
    OrderStatus,
    ResearchContext,
    StockPool,
    TradeReport,
    Universe,
)
from scheme.base import ModelAlgo, ModelParams, ModelReportForm
from scheme.config import DolphinSettings
from scheme.data.dolphindb.symbols import engine_symbol
from scheme.execute.factor import FactorAnalysisParams, analyze_factors
from scheme.execute.strategy import ResearchBacktest, StrategyAnalysisParams, run_backtest

pytestmark = pytest.mark.skipif(
    os.getenv("SOLO_TEST_DOLPHIN") != "1", reason="需要显式开启真实 DolphinDB 测试"
)


class Params(BaseModel):
    pass


class State(ResearchContext[float]):
    sent: bool = False
    fills: int = 0


class Buy(Algo[Params, State]):
    def on_snapshot(self, msg: DosVar) -> None:
        if not self.ctx.sent:
            self.backtest.submit_order(
                LimitOrder(
                    symbol="000001.XSHE",
                    time=pd.Timestamp(self.backtest.time).to_pydatetime(),
                    direction=Direction.BUY_OPEN,
                    quantity=100,
                    price=10.0,
                )
            )
            self.ctx.sent = True

    def on_trade(self, trades: Sequence[TradeReport]) -> None:
        self.ctx.fills += sum(t.tradeQty for t in trades)


class SyntheticBacktest(ResearchBacktest[State]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(["2025-01-02"]),
                "code": ["000001.XSHE"],
                "open": [10.0],
                "close": [11.0],
                "pre_close": [10.0],
                "up_limit": [20.0],
                "down_limit": [1.0],
            }
        )
        self.session.upload({"soloDaily": frame})
        self.session.run("soloMessages = soloDailyMessages(soloDaily)")
        return DosVar("soloMessages")


class CrossMonthBacktest(ResearchBacktest[State]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-03"]),
            "code": ["000001.XSHE"] * 3,
            "open": [10.0, 10.0, 10.0], "close": [11.0, 12.0, 13.0],
            "pre_close": [10.0, 11.0, 12.0], "up_limit": [20.0] * 3, "down_limit": [1.0] * 3,
        })
        frame = frame[(frame.time >= start) & (frame.time < end)]
        if frame.empty:
            self.session.run("soloMessages = table(array(TIMESTAMP,0) as timestamp)")
            return DosVar("soloMessages")
        self.session.upload({"soloDaily": frame})
        self.session.run("soloMessages = soloDailyMessages(soloDaily)")
        return DosVar("soloMessages")


def test_real_month_report_reads_only_one_portfolio_and_normalizes_costs():
    from io import StringIO

    from scheme.log import configure

    stream = StringIO()
    reads = []

    class Bounded(CrossMonthBacktest):
        def get_records(self, query, model):
            result = super().get_records(query, model)
            if query.startswith("select top 1"):
                reads.append(result)
            return result

    try:
        configure("INFO", stream=stream)
        result = run_backtest(
            [Buy(Params())], State(),
            StrategyAnalysisParams(
                start="2025-01-30", end="2025-02-04", symbols=["000001.XSHE"],
                config={"cash": "100000", "commission": "0", "tax": "0"},
            ),
            settings=DolphinSettings.from_env(), engine_type=Bounded,
        )
        assert len(reads) == 1 and len(reads[0]) == 1
        last_january = result.daily_portfolios[result.daily_portfolios.tradeDate == date(2025, 1, 31)].iloc[0]
        assert reads[0][0].tradeDate == date(2025, 1, 31)
        assert reads[0][0].netValue == pytest.approx(last_january.netValue)
        assert "2025-01 完成：" in stream.getvalue()
        assert "持仓 1 只" in stream.getvalue()

        reads.clear()
        stream.truncate(0)
        stream.seek(0)
        configure("WARNING", stream=stream)
        run_backtest(
            [Buy(Params())], State(),
            StrategyAnalysisParams(
                start="2025-01-30", end="2025-02-04", symbols=["000001.XSHE"],
                config={"cash": 100000, "commission": 0, "tax": 0},
            ),
            settings=DolphinSettings.from_env(), engine_type=Bounded,
        )
        assert reads == []
        assert stream.getvalue() == ""
    finally:
        configure("INFO")


def test_real_direct_order_and_parquet(tmp_path):
    ctx = State()
    result = run_backtest(
        [Buy(Params())],
        ctx,
        StrategyAnalysisParams(
            start="2025-01-02",
            end="2025-01-03",
            symbols=["000001.XSHE"],
            config={"cash": 100_000.0, "commission": 0.0, "tax": 0.0},
        ),
        settings=DolphinSettings.from_env(),
        engine_type=SyntheticBacktest,
    )
    assert ctx.fills == 100
    fills = result.trade_details[
        result.trade_details.orderStatus.isin([OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED])
    ]
    assert fills.tradeQty.sum() == 100
    assert result.daily_portfolios.iloc[-1].totalEquity == pytest.approx(100100.0)
    for path in result.save(tmp_path):
        pd.read_parquet(path)


def test_model_form_default_chain_with_real_matching(monkeypatch, tmp_path):
    class SignalModel(ModelAlgo[ModelParams, State]):
        def on_snapshot(self, msg: DosVar) -> None:
            if not self.ctx.sent:
                self.ctx.signal = {"000001.SZ": 1.0}
                self.ctx.sent = True

        def on_trade(self, trades: Sequence[TradeReport]) -> None:
            self.ctx.fills += sum(t.tradeQty for t in trades)

    # 没有有效历史行情，默认风险平价在求解失败后退化为等权。
    monkeypatch.setattr(
        "scheme.base.projects.optimize.default.query",
        lambda _: nullcontext(
            SimpleNamespace(data=pd.DataFrame(columns=["code", "close", "adj_factor"]))
        ),
    )
    monkeypatch.setattr(ResearchBacktest, "load_messages", SyntheticBacktest.load_messages)
    params = ModelReportForm(
        start="2025-01-02",
        end="2025-01-03",
        cash=100_000,
        commission=0,
        tax=0,
        benchmark=None,
    ).build()
    params.symbols = ["000001.XSHE"]
    state = State()
    result = params.run(SignalModel, ctx=state)
    assert state.fills == 9800
    assert state.signal is None and state.target is None and state.orders is None
    assert result.daily_portfolios.iloc[-1].totalEquity == pytest.approx(109800)
    assert result.allocation_diagnostics.method.tolist() == ["equal_weight"]
    assert len(result.save(tmp_path)) == 5


class SyntheticTickBacktest(ResearchBacktest[State]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        self.session.run("""
            raw = table(take("SZ",3) as Market, take("000001",3) as SecurityID,
                take(2025.01.02,3) as TradeDate, [09:30:00.000,09:31:00.000,15:00:00.000] as TradeTime,
                [10.,10.1,11.] as LastPrice)
            for (level in 1..5) {
                raw["BidPrice"+string(level)] = [10.,10.1,11.] - (level-1)*0.1
                raw["AskPrice"+string(level)] = [10.,10.1,11.] + (level-1)*0.1
                raw["BidVolume"+string(level)] = take(1000l,3)
                raw["AskVolume"+string(level)] = take(1000l,3)
            }
            reference = table([2025.01.02T00:00:00.000] as time, ["000001.XSHE"] as code,
                [10.] as pre_close, [20.] as up_limit, [1.] as down_limit)
            soloMessages = soloTickMessages(raw, reference, ["000001.XSHE"])
        """)
        return DosVar("soloMessages")


def test_real_tick_adapter_and_account():
    ctx = State()
    result = run_backtest(
        [Buy(Params())],
        ctx,
        StrategyAnalysisParams(
            start="2025-01-02",
            end="2025-01-03",
            symbols=["000001.XSHE"],
            config={"cash": 100000.0, "commission": 0.0, "tax": 0.0},
        ),
        settings=DolphinSettings.from_env(),
        engine_type=SyntheticTickBacktest,
    )
    assert ctx.fills == 100
    assert result.daily_portfolios.iloc[-1].totalEquity == pytest.approx(100100.0)


class TestFactor(Factor[FactorParams]):
    __test__ = False

    def compute(self, start: date, end: date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [pd.date_range(start, periods=2), ["000001.XSHE", "600000.XSHG"]],
            names=["date", "symbol"],
        )
        return pd.DataFrame({"score": [1.0, 2.0, 2.0, 1.0]}, index=index)


class SparseFactor(Factor[FactorParams]):
    __test__ = False

    def compute(self, start: date, end: date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [pd.date_range(start, periods=2), ["000001.XSHE", "000002.XSHE", "600000.XSHG"]],
            names=["date", "symbol"],
        )
        return pd.DataFrame({"score": [1.0, np.nan, 2.0, 2.0, np.nan, 1.0]}, index=index)


def query_prices(prices: pd.DataFrame):
    def query(request, *, session):
        validated = FactorQuery.model_validate(request)
        session.upload(
            {
                "testPrices": prices[
                    pd.to_datetime(prices.time).between(
                        validated.start_date, validated.end_date, inclusive="both",
                    )
                ],
                "testDefinitions": json.dumps(
                    {
                        name: node.model_dump(mode="json")
                        for name, node in validated.derivatives.items()
                    }
                ),
            }
        )
        session.run(
            "use query; testReturns = compute_factors(testPrices, fromStdJson(testDefinitions))"
        )
        return nullcontext(SimpleNamespace(data_ref="testReturns"))

    return query


@pytest.mark.parametrize("weight", ["equal", "market_value"])
def test_real_factor_statistics(monkeypatch, tmp_path, weight):
    prices = pd.DataFrame(
        {
            "time": np.repeat(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]), 2),
            "code": ["000001.SZ", "600000.SH"] * 3,
            "open": [9.0, 9.0, 5.5, 12.0, 7.5, 15.4],
            "close": [10.0, 10.0, 6.0, 14.0, 8.25, 18.2],
            "adj_factor": [1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "circ_mv": [1.0, 3.0] * 3,
        }
    )
    monkeypatch.setattr("scheme.execute.factor.api.query", query_prices(prices))
    result = analyze_factors(
        TestFactor(FactorParams(start="2025-01-02", end="2025-01-07")),
        FactorAnalysisParams(
            start="2025-01-02",
            end="2025-01-07",
            columns=["score"],
            max_lag=2,
            groups=2,
            n_select=1,
            calendar_symbol="000001.XSHE",
            weight=weight,
        ),
    )
    assert result.information_coefficient.score_r0o_ic.tolist() == pytest.approx([1.0, 1.0])
    assert result.group_returns.score_r0o_group0.tolist() == pytest.approx([0.1, 0.1])
    assert result.processed_data.score.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert result.processed_data.score_group.tolist() == [0, 1, 1, 0]
    assert result.processed_data.r0o.tolist() == pytest.approx([0.1, 0.2, 0.25, 0.1])
    assert result.processed_data.r1.tolist() == pytest.approx([12 / 11 - 1, 14 / 12 - 1, 0.1, 18.2 / 15.4 - 1])
    assert result.processed_data.r1o.iloc[:2].tolist() == pytest.approx([0.25, 0.1])
    assert result.processed_data.r2.iloc[:2].tolist() == pytest.approx([0.1, 18.2 / 15.4 - 1])
    assert result.processed_data[["r1o", "r2"]].iloc[2:].isna().all().all()
    assert set(result.group_turnover.periods) == {1, 2}
    assert result.execution_statistics.source_count.tolist() == [2, 2]
    assert result.execution_statistics.filtered_count.tolist() == [2, 2]
    assert result.processed_data.circ_mv.tolist() == (
        [1.0, 3.0] * 2 if weight == "market_value" else [1.0] * 4
    )
    for path in result.save(tmp_path):
        pd.read_parquet(path)


@pytest.mark.parametrize("missing_cap", ["000002.SZ", "600000.SH"])
def test_real_market_weight_checks_only_valid_factor_rows(monkeypatch, missing_cap):
    prices = pd.DataFrame(
        {
            "time": np.repeat(pd.date_range("2025-01-02", periods=3), 3),
            "code": ["000001.SZ", "000002.SZ", "600000.SH"] * 3,
            "open": [9.0, 8.0, 9.0, 5.0, 8.0, 11.0, 6.0, 8.0, 12.0],
            "close": [10.0, 8.0, 10.0, 5.5, 8.0, 12.0, 6.6, 8.0, 13.2],
            "adj_factor": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            "circ_mv": [1.0, 2.0, 3.0] * 3,
        }
    )
    prices.loc[prices.code == missing_cap, "circ_mv"] = np.nan
    monkeypatch.setattr("scheme.execute.factor.api.query", query_prices(prices))
    params = FactorAnalysisParams(
        start="2025-01-02",
        end="2025-01-04",
        columns=["score"],
        max_lag=1,
        groups=2,
        n_select=1,
        calendar_symbol="000001.XSHE",
        weight="market_value",
    )
    factor = SparseFactor(FactorParams(start="2025-01-02", end="2025-01-04"))
    if missing_cap == "600000.SH":
        with pytest.raises(RuntimeError, match="有效因子记录具备正数 circ_mv"):
            analyze_factors(factor, params)
        return
    result = analyze_factors(factor, params)
    assert result.processed_data.code.tolist() == ["000001.XSHE", "600000.XSHG"] * 2
    assert result.processed_data.circ_mv.tolist() == [1.0, 3.0] * 2
    assert result.execution_statistics.source_count.tolist() == [3, 3]
    assert result.execution_statistics.filtered_count.tolist() == [2, 2]


def test_real_factor_query_pipeline():
    result = analyze_factors(
        TestFactor(FactorParams(start="2026-06-01", end="2026-06-03")),
        FactorAnalysisParams(
            start="2026-06-01",
            end="2026-06-03",
            columns=["score"],
            max_lag=2,
            groups=2,
            n_select=1,
        ),
    )
    frame = result.processed_data
    assert frame.score.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert frame.score_group.tolist() == [0, 1, 1, 0]
    first = frame.time == frame.time.min()
    assert frame.loc[first, ["r0o", "r1"]].notna().all().all()
    assert frame.loc[~first, ["r0o", "r1"]].isna().all().all()
    assert frame[["r1o", "r2"]].isna().all().all()
    assert result.execution_statistics.retention_rate.tolist() == [1.0, 1.0]


def test_real_segment_returns_keep_missing_prices_sparse_factor_dates_and_end_boundary(monkeypatch):
    days = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"])
    prices = pd.DataFrame({
        "time": np.repeat(days, 2),
        "code": ["000001.SZ", "600000.SH"] * 5,
        "open": [9.0, 18.0, np.nan, 22.0, 6.0, 13.0, 7.5, 16.5, 999.0, 999.0],
        "close": [10.0, 20.0, np.nan, 24.0, 7.0, 15.0, 8.25, 18.0, 1000.0, 1000.0],
        "adj_factor": [1.0] * 4 + [2.0] * 6,
    })

    class SparseDates(Factor[FactorParams]):
        def compute(self, start: date, end: date) -> pd.DataFrame:
            index = pd.MultiIndex.from_product(
                [days[[0, 2, 3]], ["000001.XSHE", "600000.XSHG"]],
                names=["date", "symbol"],
            )
            return pd.DataFrame({"score": [1.0, 2.0, 2.0, 1.0, 1.0, 2.0]}, index=index)

    monkeypatch.setattr("scheme.execute.factor.api.query", query_prices(prices))
    params = FactorAnalysisParams(
        start="2025-01-03", end="2025-01-09", columns=["score"], max_lag=2,
        groups=2, n_select=1, calendar_symbol="600000.XSHG",
    )
    result = analyze_factors(SparseDates(FactorParams(**{
        name: getattr(params, name) for name in ("start", "end", "universe")
    })), params)
    frame = result.processed_data.set_index(["code", "time"])
    assert len(frame) == 6
    first = frame.loc[("000001.XSHE", days[0])]
    assert first[["r0o", "r1", "r1o"]].isna().all()
    assert first.r2 == pytest.approx(14 / 12 - 1)
    other = frame.loc[("600000.XSHG", days[0])]
    assert other[params.return_columns].tolist() == pytest.approx([
        22 / 20 - 1, 24 / 22 - 1, 26 / 24 - 1, 30 / 26 - 1,
    ])
    for code in ("000001.XSHE", "600000.XSHG"):
        penultimate = frame.loc[(code, days[2])]
        assert penultimate[["r0o", "r1"]].notna().all()
        assert penultimate[["r1o", "r2"]].isna().all()
        assert frame.loc[(code, days[3]), params.return_columns].isna().all()
    assert frame.loc[("000001.XSHE", days[2]), "r0o"] == pytest.approx(15 / 14 - 1)
    assert frame.loc[("000001.XSHE", days[2]), "r1"] == pytest.approx(0.1)
    assert set(result.group_turnover.periods) == {1, 2}


@pytest.mark.parametrize("pool,weight,size", [
    (StockPool.CSI300, "weight_000300SH", 300), (StockPool.SSE50, "weight_000016SH", 50),
])
def test_real_index_pool_runs_arena_two_stage_query(pool, weight, size):
    universe = Universe(pool=pool)
    start, end = date(2024, 6, 1), date(2024, 8, 1)
    panel = universe.evaluate(start, end)
    # 第一阶段只取区间候选并集：每个候选都至少入池一天，逐日成分数与指数一致。
    assert panel.any().all()
    assert (panel.sum(axis=1) == size).all()
    stacked = panel.stack()
    members = set(stacked[stacked].index)

    rank = {
        "type": "CS", "op": "unary.rank_pct", "fields": {"col": "close"},
        "params": {}, "on": Universe.member,
    }
    with universe.query(
        start, end, factors=[weight, "close"], derivatives={"member_close_rank": rank},
    ) as result:
        frame = result.data
        raw = result.session.run(f"""
            select date(time) as date, string(code) as code from {CORE_TABLE}
            where factor = `{weight}, time >= 2024.06.01, time < 2024.08.01, value > 0
        """)
    # 第二阶段不过滤：候选并集的每个交易日都有一行，成员节点作为列输出。
    assert len(frame) == panel.size
    member_rows = frame[Universe.member].eq(True)
    assert member_rows.eq(frame[weight] > 0).all()
    # on 只让当日成分参与截面排名：非成分行结果为 NULL，但不删除行。
    assert frame.member_close_rank[~member_rows].isna().all()
    assert frame.member_close_rank[member_rows & frame.close.notna()].notna().all()
    raw_members = {
        (day, engine_symbol(code))
        for day, code in zip(pd.to_datetime(raw.date), raw.code, strict=True)
        if day in panel.index
    }
    assert raw_members == members

    with universe.query(start, end, factors=[weight], filters=[Universe.member]) as result:
        frame = result.data
    assert set(zip(pd.to_datetime(frame.time), frame.code.map(engine_symbol), strict=True)) == members


def test_real_all_market_pool_has_no_first_stage():
    universe = Universe()
    assert universe.codes_query(date(2024, 7, 1), date(2024, 7, 6)) is None
    panel = universe.evaluate(date(2024, 7, 1), date(2024, 7, 6))
    assert len(panel) == 5
    assert len(panel.columns) > 4000
    assert panel.all().all()
