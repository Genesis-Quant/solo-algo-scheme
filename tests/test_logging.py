"""原有 Arena 日志契约：级别、真实来源、正文和现有前端解析器。"""

import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import date
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from backtest import Direction, OrderStatus
from loguru import logger
from pydantic import BaseModel

import scheme.log as scheme_log
from scheme import Algo, ResearchContext, StrategyAnalysisParams
from scheme.config import DolphinSettings
from scheme.execute.strategy.api import ResearchBacktest, _costs, _summarize, run_backtest
from scheme.execute.strategy.engine import Backtest

ARENA_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[\s*([A-Z]+)\s*\] "
    r"([^|]+?) \| ([^|]+?): (\d+)行\| (.*)$"
)


@pytest.fixture
def output(monkeypatch):
    stream = StringIO()
    monkeypatch.delenv(scheme_log.LEVEL_VARIABLE, raising=False)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    # 测试显式安装桥接，不让 configure 以替换 pytest 的宿主 Handler 来满足断言。
    root.handlers = [scheme_log._Intercept()]
    root.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        scheme_log.mark_simulation(None)
        scheme_log._current = None
        scheme_log.configure(stream=sys.stderr)
        root.handlers = handlers
        root.setLevel(level)


def records(output):
    parsed = []
    for line in output.getvalue().splitlines():
        match = ARENA_LINE.fullmatch(line)
        assert match is not None, line
        parsed.append(match.groups())
    return parsed


def arena(level: str, message: str) -> None:
    exec(f"logger.{level}({message!r})", {"__name__": "runtime.apps.query.api", "logger": logger})


def test_info_keeps_arena_format_and_hides_query_and_third_party_chatter(output):
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
    parsed = records(output)
    assert [row[-1] for row in parsed] == [
        "回测开始", "查询重试", "首个行情时点", "训练完成", "Notebook 记录", "连接缓慢",
    ]
    assert parsed[1][1] == "WARNING"
    assert all(row[2] != "log.py" for row in parsed)


def test_stdlib_uses_original_callsite_not_intercept_handler(output):
    scheme_log.configure(stream=output)
    logging.getLogger("model_1365.algo").info("训练完成")
    row = records(output)[0]
    assert row[2] == "test_logging.py"
    assert row[3] == "test_stdlib_uses_original_callsite_not_intercept_handler"
    assert int(row[4]) > 0


def test_debug_shows_query_steps_but_not_third_party_debug(output):
    scheme_log.configure("debug", stream=output)
    arena("info", "创建会话")
    scheme_log.get_logger("scheme.backtest").debug("逐笔明细")
    logging.getLogger("backtest").debug("批次")
    logging.getLogger("urllib3.connectionpool").debug("Starting new HTTPS connection")
    assert [row[-1] for row in records(output)] == ["创建会话", "逐笔明细", "批次"]


def test_simulation_time_is_only_a_message_prefix(output):
    scheme_log.configure(stream=output)
    log = scheme_log.get_logger("model")
    scheme_log.mark_simulation(pd.Timestamp("2023-01-03 15:00"))
    log.info("调仓")
    scheme_log.mark_simulation(None)
    log.info("报告")
    first, second = records(output)
    assert first[-1] == "[回测 2023-01-03 15:00] 调仓"
    assert second[-1] == "报告"
    assert first[2:4] == second[2:4]


def test_invalid_environment_level_falls_back_without_breaking(output, monkeypatch):
    monkeypatch.setenv(scheme_log.LEVEL_VARIABLE, "verbose")
    scheme_log.configure(stream=output)
    scheme_log.get_logger("scheme").debug("不显示")
    assert records(output)[0][-1] == "SCHEME_LOG_LEVEL 无效：VERBOSE，已按 INFO 输出日志"
    assert "不显示" not in output.getvalue()
    with pytest.raises(ValueError):
        scheme_log.configure("verbose", stream=output)


def test_warn_alias_and_exceptions_keep_traceback_and_callsite(output):
    scheme_log.configure("warn", stream=output)
    scheme_log.get_logger("scheme").info("不显示")
    try:
        raise RuntimeError("算法失败")
    except RuntimeError:
        logging.getLogger("model_1365.algo").exception("调仓失败")
    text = output.getvalue()
    assert "不显示" not in text
    header = ARENA_LINE.fullmatch(text.splitlines()[0])
    assert header is not None
    assert header.group(2) == "ERROR"
    assert header.group(3) == "test_logging.py"
    assert header.group(6) == "调仓失败"
    assert "RuntimeError: 算法失败" in text


def test_repeated_configuration_does_not_duplicate_records(output):
    scheme_log.configure("INFO", stream=output)
    scheme_log.configure("INFO", stream=output)
    logging.getLogger("backtest").info("首个行情")
    assert len(records(output)) == 1


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
    queries = []
    uploaded = {}
    engine.session = SimpleNamespace(upload=lambda values: uploaded.update(values))

    def get_records(query, model):
        queries.append(query)
        eligible = [row for row in portfolios if np.datetime64(row.tradeDate) < uploaded["soloProgressDate"]]
        return sorted(eligible, key=lambda row: row.tradeDate, reverse=True)[:1]

    engine.get_records = get_records
    engine.get_position = lambda: {
        "600000.XSHG": SimpleNamespace(longPosition=100, shortPosition=0),
        "000001.XSHE": SimpleNamespace(longPosition=0, shortPosition=0),
    }
    engine.report_month(pd.Timestamp("2023-01-03 09:30"))
    engine.progress.orders, engine.progress.trades, engine.progress.rejected = 5, 4, 1
    engine.report_month(pd.Timestamp("2023-01-31 15:00"))
    assert output.getvalue() == ""
    engine.report_month(pd.Timestamp("2023-02-01 09:30"))
    assert "2023-01 完成：净值 1.0500（本月 +5.00%），持仓 1 只，现金 1.0%；订单 5，成交 4，拒单 1；" in (
        records(output)[0][-1]
    )
    assert (engine.progress.orders, engine.progress.rejected) == (0, 0)
    assert queries == [
        "select top 1 * from getDailyTotalPortfolios(engine) "
        "where tradeDate < soloProgressDate order by tradeDate desc"
    ]
    assert uploaded["soloProgressDate"] == np.datetime64("2023-02-01", "D")


def test_month_report_skips_queries_when_info_is_hidden(output):
    scheme_log.configure("WARNING", stream=output)
    engine = object.__new__(Offline)
    engine.report_month(pd.Timestamp("2023-01-03 09:30"))
    engine.progress.orders, engine.progress.trades, engine.progress.rejected = 5, 4, 1
    # 此对象没有会话；日志不可见时不应调用任何账户或持仓查询。
    engine.report_month(pd.Timestamp("2023-02-01 09:30"))
    assert output.getvalue() == ""
    assert engine.progress.month == pd.Period("2023-02", "M")
    assert (engine.progress.orders, engine.progress.trades, engine.progress.rejected) == (0, 0, 0)
    assert engine.progress.net_value is None


def test_month_report_does_not_report_false_return_after_hidden_month(output):
    scheme_log.configure("WARNING", stream=output)
    engine = object.__new__(Offline)
    engine.report_month(pd.Timestamp("2023-01-03"))
    engine.report_month(pd.Timestamp("2023-02-01"))
    assert engine.progress.net_value is None
    engine.config = {
        "startDate": np.datetime64("2023-01-01", "D"), "endDate": np.datetime64("2023-12-31", "D"),
    }
    engine.session = SimpleNamespace(upload=lambda _: None)
    engine.get_records = lambda query, model: [
        SimpleNamespace(tradeDate=date(2023, 2, 28), netValue=0.9, cash=100.0, totalEquity=1000.0),
    ]
    engine.get_position = lambda: {}
    scheme_log.configure("INFO", stream=output)
    engine.report_month(pd.Timestamp("2023-03-01"))
    assert "本月 不可用" in records(output)[0][-1]
    assert engine.progress.net_value == 0.9


def test_month_report_counts_long_and_short_positions(output):
    scheme_log.configure(stream=output)
    engine = object.__new__(Offline)
    engine.config = {
        "startDate": np.datetime64("2023-01-01", "D"), "endDate": np.datetime64("2023-12-31", "D"),
    }
    engine.session = SimpleNamespace(upload=lambda _: None)
    engine.get_records = lambda query, model: [
        SimpleNamespace(tradeDate=date(2023, 1, 31), netValue=0.95, cash=100.0, totalEquity=1000.0),
    ]
    engine.get_position = lambda: {
        "long": SimpleNamespace(longPosition=100, shortPosition=0),
        "short": SimpleNamespace(longPosition=0, shortPosition=100),
        "both": SimpleNamespace(longPosition=100, shortPosition=100),
        "flat": SimpleNamespace(longPosition=0, shortPosition=0),
    }
    engine.report_month(pd.Timestamp("2023-01-03"))
    engine.report_month(pd.Timestamp("2023-02-01"))
    assert "持仓 3 只" in records(output)[0][-1]


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
    summary, rejections = records(output)
    assert summary[-1] == (
        "回测完成：2023-01-03 至 2023-01-05 共 3 个交易日，用时 45 分 30 秒；净值 0.9900（-1.00%），"
        "基准 +2.00%，最大回撤 10.00%，手续费 260；订单 4，全部成交 1，拒单 3"
    )
    assert rejections[1] == "WARNING"
    assert rejections[-1] == "拒单 3 笔（买入 2，卖出 1）：插件未返回原因，DEBUG 日志含逐笔明细"


@pytest.mark.parametrize("values, drawdown", [
    ([0.9], "10.00%"),
    ([0.9, 0.95], "10.00%"),
    ([1.1, 0.99], "10.00%"),
    ([0.95, 0.8], "20.00%"),
    ([1.05, 1.1], "0.00%"),
])
def test_summary_includes_initial_net_value(output, values, drawdown):
    scheme_log.configure(stream=output)
    portfolios = pd.DataFrame({
        "tradeDate": pd.date_range("2023-01-03", periods=len(values)).date,
        "netValue": values, "totalFee": 0.0,
    })
    details = pd.DataFrame(columns=["orderId", "orderStatus", "direction", "orderInfo"])
    _summarize(SimpleNamespace(daily_portfolios=portfolios, trade_details=details), 1)
    assert f"最大回撤 {drawdown}" in records(output)[0][-1]


class Params(BaseModel):
    pass


class State(ResearchContext[float]):
    pass


class Probe(Algo[Params, State]):
    pass


class NoMarket(ResearchBacktest[State]):
    def __init__(self, settings, parameters, ctx, algos):
        self.config = _costs(parameters.config)

    def run(self, batch):
        pass

    def close(self):
        pass

    def get_daily_total_portfolios(self):
        return []

    def get_trade_details(self):
        return []

    def get_daily_position(self):
        return []

    def get_daily_trading_statistics(self):
        return []


@pytest.mark.parametrize("level", ["INFO", "DEBUG", "WARNING"])
def test_numeric_string_costs_do_not_fail_before_engine_creation(output, level):
    scheme_log.configure(level, stream=output)
    config = {"cash": "100000", "commission": "0.0003", "tax": "0.0005"}
    params = StrategyAnalysisParams(
        start="2023-01-03", end="2023-01-05", symbols=["000001.XSHE"], config=config,
    )
    result = run_backtest(
        [Probe(Params())], State(), params, settings=DolphinSettings.model_construct(),
        engine_type=NoMarket,
    )
    assert result.daily_portfolios.empty
    assert params.config == config
    if level in {"INFO", "DEBUG"}:
        assert "资金 100,000，佣金 0.0300%，印花税 0.0500%" in output.getvalue()
    assert _costs(config) == {"cash": 100000.0, "commission": 0.0003, "tax": 0.0005}
    assert _costs({}) == {"cash": 1_000_000.0, "commission": 0.0003, "tax": 0.0005}


def test_existing_frontend_parser_reads_logs_without_format_changes(output):
    parser = Path(__file__).resolve().parents[3] / "frontend/src/assets/lib/taskLogs.ts"
    if not parser.is_file() or shutil.which("node") is None:
        pytest.skip("消费端契约测试需要 Solo 工作区和 Node.js")
    scheme_log.configure(stream=output)
    messages = [
        "回测开始：资金 1,000,000",
        "2026-06 完成：净值 0.9199（本月 -8.01%），持仓 28 只",
        "回测完成：净值 0.9226（-7.74%），基准 -0.66%，最大回撤 8.45%",
        "拒单 7 笔（买入 7，卖出 0）：插件未返回原因",
        "消息含分隔符 | 和 - 原样保留",
    ]
    log = scheme_log.get_logger("scheme.backtest")
    log.info(messages[0])
    scheme_log.mark_simulation(pd.Timestamp("2026-07-01 09:30"))
    log.info(messages[1])
    scheme_log.mark_simulation(None)
    log.info(messages[2])
    logging.getLogger("backtest").warning(messages[3])
    log.info(messages[4])
    expected = messages.copy()
    expected[1] = "[回测 2026-07-01 09:30] " + expected[1]
    script = """
        import { readFileSync } from 'node:fs';
        const { parseTaskLog } = await import(process.argv[1]);
        const text = readFileSync(0, 'utf8');
        const lines = text.trimEnd().split('\\n');
        const direct = parseTaskLog(text);
        const wrapped = parseTaskLog(lines.map(x => '[INFO] 2026-10-11 01:00:00 +0800 - -> '+x).join('\\n'));
        process.stdout.write(JSON.stringify({direct, wrapped}));
    """
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script, parser.as_uri()],
        input=output.getvalue(), text=True, encoding="utf-8", capture_output=True, check=True,
    )
    for parsed in json.loads(result.stdout).values():
        assert [row["message"] for row in parsed] == expected
        assert [row["level"] for row in parsed] == ["INFO", "INFO", "INFO", "WARNING", "INFO"]
        assert all(row["format"] == "loguru" for row in parsed)
        assert all(row["file"] == "test_logging.py" and row["sourceLine"] > 0 for row in parsed)
        assert all(row["timestamp"] and row["source"] for row in parsed)
