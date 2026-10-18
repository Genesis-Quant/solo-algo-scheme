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
    columns = ["score"]

    def compute(self, start: date, end: date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [pd.date_range(start, periods=2), ["000001.XSHE", "600000.XSHG"]],
            names=["date", "symbol"],
        )
        return pd.DataFrame({"score": [1.0, 2.0, 2.0, 1.0]}, index=index)


class SparseFactor(Factor[FactorParams]):
    __test__ = False
    columns = ["score"]

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
            "time": np.repeat(pd.to_datetime([
                "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
            ]), 2),
            "code": ["000001.SZ", "600000.SH"] * 4,
            "open_hfq": [9.0, 9.0, 11.0, 12.0, 15.0, 15.4, 18.0, 20.0],
            "circ_mv": [1.0, 3.0] * 4,
        }
    )
    monkeypatch.setattr("scheme.execute.factor.api.query", query_prices(prices))
    result = analyze_factors(
        TestFactor(FactorParams()),
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
    assert result.information_coefficient.score_P1L1_ic.tolist() == pytest.approx([-1.0, -1.0])
    assert result.group_returns.time.tolist() == pd.DatetimeIndex(prices.time.unique()).tolist()
    assert result.group_returns.score_P1L1_group0.iloc[:2].tolist() == pytest.approx([
        np.log(15 / 11), np.log(20 / 15.4),
    ])
    assert result.group_returns.drop(columns="time").iloc[2:].isna().all().all()
    assert result.processed_data.score.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert result.processed_data.score_group.tolist() == [0, 1, 1, 0]
    assert result.processed_data.P1L1.tolist() == pytest.approx([
        np.log(15 / 11), np.log(15.4 / 12), np.log(18 / 15), np.log(20 / 15.4),
    ])
    assert result.processed_data.P1L2.iloc[:2].tolist() == pytest.approx([
        np.log(18 / 15), np.log(20 / 15.4),
    ])
    assert result.processed_data.P1L2.iloc[2:].isna().all()
    assert set(result.group_turnover.periods) == {1}
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
            "open_hfq": [9.0, 8.0, 9.0, 10.0, 8.0, 11.0, 12.0, 8.0, 12.0],
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
    factor = SparseFactor(FactorParams())
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
        TestFactor(FactorParams()),
        FactorAnalysisParams(
            start="2026-06-01",
            end="2026-06-02",
            columns=["score"],
            max_lag=2,
            groups=2,
            n_select=1,
        ),
    )
    frame = result.processed_data
    assert frame.score.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert frame.score_group.tolist() == [0, 1, 1, 0]
    assert frame[["P1L1", "P1L2"]].isna().all().all()
    assert "r0o" not in frame.columns
    assert result.execution_statistics.retention_rate.tolist() == [1.0, 1.0]


def test_real_holding_returns_keep_missing_prices_sparse_factor_dates_and_end_boundary(monkeypatch):
    days = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"])
    prices = pd.DataFrame({
        "time": np.repeat(days, 2),
        "code": ["000001.SZ", "600000.SH"] * 5,
        "open_hfq": [9.0, 18.0, np.nan, 22.0, 12.0, 26.0, 15.0, 33.0, 999.0, 999.0],
    })

    class SparseDates(Factor[FactorParams]):
        columns = ["score"]

        def compute(self, start: date, end: date) -> pd.DataFrame:
            index = pd.MultiIndex.from_product(
                [days[[0, 2, 3]], ["000001.XSHE", "600000.XSHG"]],
                names=["date", "symbol"],
            )
            return pd.DataFrame({"score": [1.0, 2.0, 2.0, 1.0, 1.0, 2.0]}, index=index)

    monkeypatch.setattr("scheme.execute.factor.api.query", query_prices(prices))
    params = FactorAnalysisParams(
        start="2025-01-03", end="2025-01-08", columns=["score"], max_lag=2,
        groups=2, n_select=1, calendar_symbol="600000.XSHG",
    )
    result = analyze_factors(SparseDates(FactorParams(universe=params.universe)), params)
    frame = result.processed_data.set_index(["code", "time"])
    assert len(frame) == 6
    first = frame.loc[("000001.XSHE", days[0])]
    assert pd.isna(first.P1L1)
    assert first.P1L2 == pytest.approx(np.log(15 / 12))
    other = frame.loc[("600000.XSHG", days[0])]
    assert other[params.return_columns].tolist() == pytest.approx([np.log(26 / 22), np.log(33 / 26)])
    for code in ("000001.XSHE", "600000.XSHG"):
        assert frame.loc[(code, days[2]), params.return_columns].isna().all()
        assert frame.loc[(code, days[3]), params.return_columns].isna().all()
    assert set(result.group_turnover.periods) == {1}


@pytest.mark.parametrize("period", [1, 5])
@pytest.mark.parametrize("weight", ["equal", "market_value"])
def test_real_log_groups_and_ic_keep_raw_labels_and_full_sparse_calendar(monkeypatch, period, weight):
    days = pd.bdate_range("2025-01-03", periods=11)
    codes = ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]
    source_codes = [engine_symbol(code) for code in codes]
    values = np.array([
        [10, 11, 13, 9, 18, 8, 21, np.nan, 30, 17, 35],
        [20, 18, 19, 22, 17, 28, 25, 39, 31, 45, 40],
        [40, 45, 50, 55, 60, 62, 58, 70, 68, 77, 83],
        [15, 16, 17, 18, 20, 19, 18, 20, 22, 30, 31],
    ], dtype=float)
    caps = np.array([1.0, 3.0, 2.0, 5.0])
    factor_values = {"score": np.array([1.0, 2.0, 3.0, 4.0]), "quality": np.array([4.0, 1.0, 3.0, 2.0])}
    prices = pd.DataFrame({
        "time": np.repeat(days, len(codes)), "code": codes * len(days),
        "open_hfq": values.T.ravel(), "circ_mv": np.tile(caps, len(days)),
    })
    valid_positions = [1, 2, 4, 7]
    observed_positions = [1, 2, 3, 4, 7, 10]

    class SparseCalendarFactor(Factor[FactorParams]):
        columns = ["score", "quality"]

        def compute(self, start: date, end: date) -> pd.DataFrame:
            index = pd.MultiIndex.from_product(
                [days[observed_positions], source_codes], names=["date", "symbol"],
            )
            panel = pd.DataFrame({
                name: np.tile(factor_values[name], len(observed_positions))
                for name in self.columns
            }, index=index)
            # These observed dates have no common valid factor sample; other dates have no rows.
            panel.loc[(days[[3, 10]], slice(None)), "quality"] = np.nan
            return panel

    monkeypatch.setattr("scheme.execute.factor.api.query", query_prices(prices))
    params = FactorAnalysisParams(
        start=days[0].date(), end=days[-1].date(), period=period, max_lag=3,
        groups=2, n_select=2, calendar_symbol=source_codes[1], weight=weight,
    )
    result = analyze_factors(SparseCalendarFactor(FactorParams()), params)
    expected_groups = pd.DataFrame(index=pd.DatetimeIndex(days, name="time"))
    expected_ic = pd.DataFrame(index=pd.DatetimeIndex(days[valid_positions], name="time"))
    weights = caps if weight == "market_value" else np.ones(len(codes))
    for factor_name, scores in factor_values.items():
        order = np.argsort(scores)
        portfolios = {"bottom": order[:2], "group0": order[:2], "group1": order[2:], "top": order[-2:]}
        for lag, ret_name in enumerate(params.return_columns, start=1):
            for portfolio in portfolios:
                expected_groups[f"{factor_name}_{ret_name}_{portfolio}"] = np.nan
            expected_ic[f"{factor_name}_{ret_name}_ic"] = np.nan
            expected_ic[f"{factor_name}_{ret_name}_rank_ic"] = np.nan
            for position in valid_positions:
                day = days[position]
                labels = np.full(len(codes), np.nan)
                entry, exit_position = position + lag, position + lag + period
                if exit_position < len(days):
                    positive = (values[:, entry] > 0) & (values[:, exit_position] > 0)
                    labels[positive] = np.log(values[positive, exit_position] / values[positive, entry])
                for portfolio, members in portfolios.items():
                    valid = members[np.isfinite(labels[members])]
                    if len(valid):
                        expected_groups.loc[day, f"{factor_name}_{ret_name}_{portfolio}"] = np.average(
                            labels[valid], weights=weights[valid],
                        )
                valid = np.isfinite(labels)
                if valid.sum() >= 2:
                    sample_scores = pd.Series(scores[valid])
                    sample_labels = pd.Series(labels[valid])
                    expected_ic.loc[day, f"{factor_name}_{ret_name}_ic"] = sample_scores.corr(sample_labels)
                    expected_ic.loc[day, f"{factor_name}_{ret_name}_rank_ic"] = sample_scores.rank().corr(
                        sample_labels.rank(),
                    )
    actual_groups = result.group_returns.set_index("time")
    pd.testing.assert_frame_equal(actual_groups, expected_groups, check_freq=False)
    pd.testing.assert_frame_equal(
        result.information_coefficient.set_index("time"), expected_ic, check_freq=False,
    )
    missing_days = days[[0, 3, 5, 6, 8, 9, 10]]
    assert actual_groups.loc[missing_days].isna().all().all()
    # Group returns are an arithmetic/cap-weighted mean of raw log labels, not log of gross mean.
    labels = np.log(values[:2, 2 + 1 + period] / values[:2, 2 + 1])
    group_return = actual_groups.loc[days[2], f"score_P{period}L1_group0"]
    assert group_return == pytest.approx(np.average(labels, weights=weights[:2]))
    assert group_return != pytest.approx(np.log(np.average(np.exp(labels), weights=weights[:2])))
    if period > 1:
        assert group_return != pytest.approx(np.average(labels, weights=weights[:2]) / period)


@pytest.mark.parametrize("price_column", ["open", "close", "open_hfq", "close_hfq"])
def test_real_p5_log_returns_compile_all_lags_and_reject_nonpositive_endpoints(
    monkeypatch, price_column,
):
    days = pd.bdate_range("2025-01-03", periods=12)
    calendar = days[:-1]
    endpoints = [
        (10.0, 12.0), (np.nan, 12.0), (10.0, np.nan), (np.nan, np.nan),
        (0.0, 12.0), (10.0, 0.0), (0.0, 0.0),
        (-10.0, 12.0), (10.0, -12.0), (-10.0, -12.0),
    ]
    cases = [(lag, entry, exit_price) for lag in range(1, 4) for entry, exit_price in endpoints]
    codes = [f"{i + 1:06d}.SZ" for i in range(len(cases))]
    source_codes = [engine_symbol(code) for code in codes]
    rows = []
    for i, (code, (lag, entry, exit_price)) in enumerate(zip(codes, cases, strict=True)):
        positions = np.arange(len(days), dtype=float)
        quotes = pd.DataFrame({
            "time": days, "code": code,
            "open": 100 + 3 * positions + i,
            "close": 200 + positions ** 2 + i,
            "open_hfq": 50 + 2 * positions ** 2 + i,
            "close_hfq": 70 + 3 * positions ** 2 + i,
            "adj_factor": 7 + 5 * positions,
        })
        quotes.loc[0, price_column] = -999.0  # The signal-day quote is never an endpoint.
        quotes.loc[lag, price_column] = entry
        quotes.loc[lag + 5, price_column] = exit_price
        quotes.loc[len(days) - 1, price_column] = 999999.0  # Outside the closed query interval.
        rows.append(quotes)
    prices = pd.concat(rows, ignore_index=True)

    class EndpointFactor(Factor[FactorParams]):
        columns = ["score"]

        def compute(self, start: date, end: date) -> pd.DataFrame:
            index = pd.MultiIndex.from_product(
                [calendar, source_codes], names=["date", "symbol"],
            )
            return pd.DataFrame({"score": np.tile(np.arange(len(codes)) + 1.0, len(calendar))}, index=index)

    monkeypatch.setattr("scheme.execute.factor.api.query", query_prices(prices))
    params = FactorAnalysisParams(
        start=calendar[0].date(), end=calendar[-1].date(), period=5, max_lag=3,
        price_column=price_column, groups=2, n_select=2, calendar_symbol=source_codes[0],
    )
    result = analyze_factors(EndpointFactor(FactorParams()), params)
    actual = result.processed_data.set_index(["code", "time"])[params.return_columns].sort_index()
    expected_rows = []
    for code, source_code in zip(codes, source_codes, strict=True):
        values = prices.loc[(prices.code == code) & prices.time.isin(calendar), price_column].to_numpy()
        for position, day in enumerate(calendar):
            row = {"code": source_code, "time": day}
            for lag, name in enumerate(params.return_columns, start=1):
                entry_position, exit_position = position + lag, position + lag + params.period
                label = np.nan
                if exit_position < len(calendar):
                    entry, exit_price = values[entry_position], values[exit_position]
                    if entry > 0 and exit_price > 0:
                        label = np.log(exit_price / entry)
                row[name] = label
            expected_rows.append(row)
    expected = pd.DataFrame(expected_rows).set_index(["code", "time"]).sort_index()
    pd.testing.assert_frame_equal(actual, expected)
    assert not np.isinf(actual).any().any()
    for code, (lag, entry, exit_price) in zip(source_codes, cases, strict=True):
        label = actual.loc[(code, calendar[0]), f"P5L{lag}"]
        if entry > 0 and exit_price > 0:
            assert label == pytest.approx(np.log(exit_price / entry))
        else:
            assert pd.isna(label)
    assert result.group_returns.time.tolist() == calendar.tolist()
    assert {spec["kind"] for spec in params.return_specs.values()} == {"log"}


@pytest.mark.parametrize("pool,weight,size", [
    (StockPool.CSI300, "weight_000300SH", 300), (StockPool.SSE50, "weight_000016SH", 50),
])
def test_real_index_pool_runs_one_query_over_fixed_codes(pool, weight, size):
    universe = Universe(pool=pool)
    start, end = date(2024, 6, 1), date(2024, 8, 1)
    panel = universe.evaluate(start, end)
    # 全时段证券域保留区间外代码，不要求每个代码在当前区间内入池。
    assert not panel.any().all()
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
    # 单阶段不过滤：固定证券域的每个真实交易日都有一行，成员节点作为列输出。
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


def test_real_all_market_pool_uses_fixed_process_codes():
    universe = Universe()
    assert not hasattr(universe, "codes_query")
    assert universe.dataset_query(date(2024, 7, 1), date(2024, 7, 6)).codes == []
    panel = universe.evaluate(date(2024, 7, 1), date(2024, 7, 6))
    assert len(panel) == 5
    assert len(panel.columns) > 4000
    assert panel.all().all()
