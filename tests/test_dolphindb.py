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

from scheme import (
    Algo,
    Direction,
    DosVar,
    Factor,
    FactorParams,
    LimitOrder,
    OrderStatus,
    ResearchContext,
    TradeReport,
)
from scheme.apps.backtest import BacktestParameters, ResearchBacktest, run_backtest
from scheme.apps.factor import FactorAnalysisParameters, analyze_factors
from scheme.config import DolphinSettings

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


def test_real_direct_order_and_parquet(tmp_path):
    ctx = State()
    result = run_backtest(
        [Buy(Params())],
        ctx,
        BacktestParameters(
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
        BacktestParameters(
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


@pytest.mark.parametrize("weight", ["equal", "market_value"])
def test_real_factor_statistics(monkeypatch, tmp_path, weight):
    prices = pd.DataFrame(
        {
            "time": np.repeat(pd.date_range("2025-01-02", periods=3), 2),
            "code": ["000001.SZ", "600000.SH"] * 3,
            "close": [10.0, 10.0, 5.5, 12.0, 6.6, 13.2],
            "adj_factor": [1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "circ_mv": [1.0, 3.0] * 3,
        }
    )

    def query_prices(request, *, session):
        validated = FactorQuery.model_validate(request)
        session.upload(
            {
                "testPrices": prices,
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

    monkeypatch.setattr("scheme.apps.factor.api.query", query_prices)
    result = analyze_factors(
        TestFactor(FactorParams(start="2025-01-02", end="2025-01-04")),
        FactorAnalysisParameters(
            start="2025-01-02",
            end="2025-01-04",
            columns=["score"],
            return_periods=[1, 2],
            groups=2,
            n_select=1,
            calendar_symbol="000001.XSHE",
            weight=weight,
        ),
    )
    assert result.information_coefficient.score_return_1_ic.tolist() == pytest.approx([1.0, 1.0])
    assert result.group_returns.score_return_1_group0.tolist() == pytest.approx([0.1, 0.1])
    assert result.processed_data.score.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert result.processed_data.score_group.tolist() == [0, 1, 1, 0]
    assert result.processed_data.return_1.tolist() == pytest.approx([0.1, 0.2, 0.2, 0.1])
    assert result.processed_data.return_2.iloc[:2].tolist() == pytest.approx([0.32, 0.32])
    assert result.processed_data.return_2.iloc[2:].isna().all()
    assert result.execution_statistics.source_count.tolist() == [2, 2]
    assert result.execution_statistics.filtered_count.tolist() == [2, 2]
    assert result.processed_data.circ_mv.tolist() == (
        [1.0, 3.0] * 2 if weight == "market_value" else [1.0] * 4
    )
    for path in result.save(tmp_path):
        pd.read_parquet(path)


def test_real_factor_query_pipeline():
    result = analyze_factors(
        TestFactor(FactorParams(start="2026-06-01", end="2026-06-03")),
        FactorAnalysisParameters(
            start="2026-06-01",
            end="2026-06-03",
            columns=["score"],
            return_periods=[1, 5],
            groups=2,
            n_select=1,
        ),
    )
    assert result.processed_data.score.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert result.processed_data.score_group.tolist() == [0, 1, 1, 0]
    assert result.processed_data[["return_1", "return_5"]].notna().all().all()
    assert result.execution_statistics.retention_rate.tolist() == [1.0, 1.0]
