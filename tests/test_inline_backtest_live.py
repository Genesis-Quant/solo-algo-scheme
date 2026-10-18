"""Execute the literal callback adapter against the real Backtest plugin."""

import os
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from scheme.config import DolphinSettings
from scheme.data.dolphindb.backtest import _InlineBacktestSession
from scheme.data.dolphindb.database import create_session

pytestmark = pytest.mark.skipif(
    os.getenv("SOLO_TEST_DOLPHIN") != "1", reason="需要显式开启真实 DolphinDB 测试",
)


def test_real_inline_daily_roots_and_lazy_return_summary():
    market = pd.DataFrame({
        "time": np.asarray(["2025-01-02", "2025-01-03"], dtype="datetime64[ms]"),
        "code": ["000001.XSHE"] * 2,
        "open": [10.0, 11.0], "close": [11.0, 12.0], "low": [9.0, 10.0],
        "high": [12.0, 13.0], "pre_close": [9.0, 11.0],
        "up_limit": [20.0, 20.0], "down_limit": [1.0, 1.0],
    })
    with create_session(DolphinSettings.from_env()) as session:
        proxy = _InlineBacktestSession(session)
        proxy.run("use backtest")
        proxy.upload({
            "coreBacktestData": market,
            "coreBacktestAdj": "",
            "coreBacktestSyntheticSpread": 0.0,
            "coreBacktestName": "soloInlineTest" + uuid4().hex,
            "coreBacktestConfig": {
                "startDate": np.datetime64("2025-01-02", "D"),
                "endDate": np.datetime64("2025-01-03", "D"),
                "strategyGroup": "stock", "cash": 100000.0,
                "commission": 0.0, "tax": 0.0,
                "msgAsTable": True, "msgAsPiecesOnSnapshot": True,
            },
            "coreBacktestAnnualTradingDays": np.int32(252),
            "coreBacktestRiskFreeRate": 0.0,
        })
        proxy.run("""
            coreBacktestMessage = backtest::build_backtest_message(
                coreBacktestData, coreBacktestAdj, coreBacktestSyntheticSpread)
            def initialize(mutable context) { context["snapshots"] = 0l }
            def beforeTrading(mutable context) {}
            def onBar(mutable context, msg, indicator) {}
            def onSnapshot(mutable context, msg, indicator) {
                context["snapshots"] += 1l
            }
            def onOrder(mutable context, orders) {}
            def onTrade(mutable context, trades) {}
            def afterTrading(mutable context) {}
            def finalize(mutable context) {}
            coreBacktestEngine = backtest::run_backtest(
                coreBacktestName, coreBacktestConfig, coreBacktestMessage,
                initialize, beforeTrading, onBar, onSnapshot, onOrder, onTrade,
                afterTrading, finalize)
        """)
        try:
            assert proxy.run("coreBacktestMessage.rows()") == 4
            assert proxy.run('Backtest::getContextDict(coreBacktestEngine)["snapshots"]') >= 4
            portfolios = proxy.run("Backtest::getDailyTotalPortfolios(coreBacktestEngine)")
            assert len(portfolios) == 2
            assert portfolios.netValue.eq(1).all()
            summary = proxy.run("""
                backtest::standardize_return_summary(
                    Backtest::getReturnSummary(coreBacktestEngine),
                    Backtest::getDailyTotalPortfolios(coreBacktestEngine),
                    coreBacktestAnnualTradingDays, coreBacktestRiskFreeRate)
            """)
            assert len(summary) == 1
            assert summary.annualReturn.iloc[0] == pytest.approx(0.0)
        finally:
            proxy.run("Backtest::dropBacktestEngine(coreBacktestEngine)")
