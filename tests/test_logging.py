"""统一日志：级别、来源过滤、模拟时间和回测汇总，不连接 DolphinDB。"""

import logging
import sys
from datetime import date
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from backtest import Direction, OrderStatus
from loguru import logger

import scheme.log as scheme_log
from scheme.execute.strategy.api import _summarize
from scheme.execute.strategy.engine import Backtest


@pytest.fixture
def output(monkeypatch):
    stream = StringIO()
    monkeypatch.delenv(scheme_log.LEVEL_VARIABLE, raising=False)
    yield stream
    scheme_log.mark_simulation(None)
    scheme_log._current = None
    scheme_log.configure(stream=sys.stderr)


def arena(level: str, message: str) -> None:
    # loguru 按调用方模块的 __name__ 识别来源，模拟 Arena 查询模块的逐步日志。
    exec(f"logger.{level}({message!r})", {"__name__": "runtime.apps.query.api", "logger": logger})


def test_info_hides_query_steps_and_third_party_chatter(output):
    scheme_log.configure("INFO", stream=output)
    scheme_log.get_logger("scheme.backtest").info("回测开始")
    scheme_log.get_logger("scheme.backtest").debug("逐笔明细")
    arena("info", "创建会话")
    arena("warning", "查询重试")
    logging.getLogger("backtest").info("首个行情时点")
    logging.getLogger("model_1365.algo").info("训练完成")
    logging.getLogger("__main__").info("Notebook 记录")
    logging.getLogger("httpx").info("HTTP Request")
    logging.getLogger("httpx").warning("连接缓慢")
    lines = output.getvalue().splitlines()
    assert [line.split(" | ", 2)[2] for line in lines] == [
        "scheme.backtest | 回测开始",
        "arena.query | 查询重试",
        "backtest | 首个行情时点",
        "model_1365.algo | 训练完成",
        "__main__ | Notebook 记录",
        "httpx | 连接缓慢",
    ]
    assert lines[1].split(" | ")[1].strip() == "WARNING"


def test_debug_shows_query_steps_but_not_third_party_debug(output):
    scheme_log.configure("debug", stream=output)
    arena("info", "创建会话")
    scheme_log.get_logger("scheme.backtest").debug("逐笔明细")
    logging.getLogger("backtest").debug("批次")
    logging.getLogger("urllib3.connectionpool").debug("Starting new HTTPS connection")
    text = output.getvalue()
    assert "arena.query | 创建会话" in text
    assert "scheme.backtest | 逐笔明细" in text
    assert "backtest | 批次" in text
    assert "Starting new HTTPS connection" not in text


def test_simulation_time_marks_lines_until_cleared(output):
    scheme_log.configure(stream=output)
    log = scheme_log.get_logger("model")
    scheme_log.mark_simulation(pd.Timestamp("2023-01-03 15:00"))
    log.info("调仓")
    scheme_log.mark_simulation(None)
    log.info("报告")
    first, second = output.getvalue().splitlines()
    assert first.endswith("| model | 2023-01-03 15:00 | 调仓")
    assert second.endswith("| model | 报告")


def test_invalid_environment_level_falls_back_without_breaking(output, monkeypatch):
    monkeypatch.setenv(scheme_log.LEVEL_VARIABLE, "verbose")
    scheme_log.configure(stream=output)
    scheme_log.get_logger("scheme").debug("不显示")
    assert "SCHEME_LOG_LEVEL 无效：VERBOSE，已按 INFO 输出日志" in output.getvalue()
    assert "不显示" not in output.getvalue()
    with pytest.raises(ValueError):
        scheme_log.configure("verbose", stream=output)


def test_warn_alias_and_exceptions_keep_traceback(output):
    scheme_log.configure("warn", stream=output)
    scheme_log.get_logger("scheme").info("不显示")
    try:
        raise RuntimeError("算法失败")
    except RuntimeError:
        logging.getLogger("model_1365.algo").exception("调仓失败")
    text = output.getvalue()
    assert "不显示" not in text
    assert "ERROR   | model_1365.algo | 调仓失败" in text
    assert "RuntimeError: 算法失败" in text


def test_duration_is_readable():
    assert scheme_log.duration(42.9) == "42 秒"
    assert scheme_log.duration(2730) == "45 分 30 秒"
    assert scheme_log.duration(7380) == "2 小时 3 分"


class Offline(Backtest):
    def load_messages(self, start, end):
        raise NotImplementedError


def test_month_report_uses_previous_month_end(output):
    scheme_log.configure(stream=output)
    engine = object.__new__(Offline)
    engine.config = {
        "startDate": np.datetime64("2023-01-01", "D"), "endDate": np.datetime64("2023-12-31", "D"),
    }
    portfolios = [
        SimpleNamespace(tradeDate=date(2023, 1, 30), netValue=1.02, cash=500.0, totalEquity=1e5),
        SimpleNamespace(tradeDate=date(2023, 1, 31), netValue=1.05, cash=1050.0, totalEquity=1.05e5),
        SimpleNamespace(tradeDate=date(2023, 2, 1), netValue=9.99, cash=0.0, totalEquity=1.0),
    ]
    engine.get_daily_total_portfolios = lambda: portfolios
    engine.get_position = lambda: {
        "600000.XSHG": SimpleNamespace(longPosition=100),
        "000001.XSHE": SimpleNamespace(longPosition=0),
    }
    engine.report_month(pd.Timestamp("2023-01-03 09:30"))
    engine.progress.orders, engine.progress.trades, engine.progress.rejected = 5, 4, 1
    engine.report_month(pd.Timestamp("2023-01-31 15:00"))
    assert output.getvalue() == ""
    engine.report_month(pd.Timestamp("2023-02-01 09:30"))
    assert "2023-01 完成：净值 1.0500（本月 +5.00%），持仓 1 只，现金 1.0%；订单 5，成交 4，拒单 1；" in (
        output.getvalue()
    )
    assert (engine.progress.orders, engine.progress.rejected) == (0, 0)


def test_summary_reports_returns_orders_and_rejections(output):
    scheme_log.configure(stream=output)
    portfolios = pd.DataFrame({
        "tradeDate": [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
        "netValue": [1.0, 1.1, 0.99], "totalFee": [0.0, 120.0, 260.5],
        "benchmarkNetValue": [1.0, 1.01, 1.02],
    })
    details = pd.DataFrame({
        "orderId": [1, 1, 2, 2, 3, 3, 4, 4],
        "direction": [Direction.BUY_OPEN] * 4 + [Direction.BUY_OPEN] * 2 + [Direction.SELL_CLOSE] * 2,
        "orderStatus": [
            OrderStatus.REPORTED, OrderStatus.FILLED, OrderStatus.REPORTED, OrderStatus.REJECTED,
            OrderStatus.REPORTED, OrderStatus.REJECTED, OrderStatus.REPORTED, OrderStatus.REJECTED,
        ],
        "orderInfo": [None, None, None, None, None, "", None, None],
    })
    _summarize(SimpleNamespace(daily_portfolios=portfolios, trade_details=details), 2730)
    summary, rejections = output.getvalue().splitlines()
    assert summary.endswith(
        "回测完成：2023-01-03 至 2023-01-05 共 3 个交易日，用时 45 分 30 秒；净值 0.9900（-1.00%），"
        "基准 +2.00%，最大回撤 10.00%，手续费 260；订单 4，全部成交 1，拒单 3"
    )
    assert "WARNING" in rejections
    assert rejections.endswith("拒单 3 笔（买入 2，卖出 1）：插件未返回原因，DEBUG 日志含逐笔明细")
