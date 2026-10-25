from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pydantic import BaseModel

from scheme.base import Algo, StrategyAnalysisParams
from scheme.base.projects.execution.default import DirectExecution
from scheme.base.projects.optimize.default import RiskParity
from scheme.data.dolphindb import api as queries
from scheme.execute.strategy import api
from scheme.execute.strategy.engine import Backtest


class State(BaseModel):
    pass


class ExternalParams(BaseModel):
    lookback_days: str = "provider-owned, not a frame count"


class ExternalAlgo(Algo[ExternalParams, BaseModel]):
    def __init__(self):
        super().__init__(ExternalParams())
        self.events = []

    def initialize(self):
        self.events.append("initialize")

    def on_snapshot(self, msg):
        self.events.append("snapshot")

    def on_order(self, orders):
        self.events.append("order")

    def on_trade(self, trades):
        self.events.append("trade")


def parameters(**changes):
    values = dict(
        start=date(2025, 1, 1),
        end=date(2025, 1, 10),
        symbols=["600000.SH"],
        benchmark=None,
    )
    values.update(changes)
    return StrategyAnalysisParams(**values)


def test_research_adapter_does_not_inspect_algo_parameters_or_query_universe(monkeypatch):
    algo = ExternalAlgo()

    def init(self, settings, **kwargs):
        self.session = Mock()
        self.algos = kwargs["algos"]
        self.time = np.datetime64("NaT", "ms")

    monkeypatch.setattr(Backtest, "__init__", init)
    monkeypatch.setattr(api, "query", Mock(side_effect=AssertionError("no research preload")))
    engine = api.ResearchBacktest(Mock(), parameters(), State(), [algo])
    assert engine.symbols == ["600000.XSHG"]
    assert "universe" not in engine.__dict__
    assert not hasattr(engine, "history")
    assert algo.params.lookback_days == "provider-owned, not a frame count"


def test_custom_engine_does_not_need_history_or_adapter_private_methods():
    algo = ExternalAlgo()
    events = []

    class IndependentEngine:
        def __init__(self, settings, parameters, ctx, algos):
            self.session = object()
            self.algos = algos

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

        def run(self, batch_size):
            events.append(batch_size)
            for item in self.algos:
                item.initialize()
                item.on_snapshot(None)

        def get_trade_details(self):
            return []

        get_daily_position = get_trade_details
        get_daily_total_portfolios = get_trade_details
        get_daily_trading_statistics = get_trade_details

    result = api.execute_backtest([algo], State(), parameters(), Mock(), IndependentEngine)
    assert result.daily_portfolios.empty
    assert algo.events == ["initialize", "snapshot"]
    assert events == [np.timedelta64(1, "D"), "closed"]


def test_query_uses_only_explicit_source_or_default_settings(monkeypatch):
    default = SimpleNamespace(database="dfs://default", table="prices", password=Mock())
    explicit = SimpleNamespace(database="dfs://independent", table="bars", password=Mock())
    session = object()
    from_env = Mock(return_value=default)
    monkeypatch.setattr(queries.DolphinSettings, "from_env", from_env)
    monkeypatch.setattr(queries.query_api, "available_factors", lambda: ["close"])
    result = Mock()
    factory = Mock(return_value=result)
    monkeypatch.setattr(queries, "QueryResult", factory)
    request = {
        "start_date": "2025-01-02",
        "end_date": "2025-01-03",
        "codes": ["600000.SH"],
        "factors": ["close"],
    }
    for source, connection in ((None, None), (explicit, session)):
        actual = default if source is None else source
        assert queries.query(request, settings=source, session=connection) is result
        assert factory.call_args.kwargs["session"] is connection
        assert factory.call_args.kwargs["settings"] is actual
        assert factory.call_args.args[0]["database"] == actual.database
        assert factory.call_args.args[0]["table"] == actual.table
    from_env.assert_called_once_with()


def test_account_reads_are_live_and_no_session_proxy():
    session = Mock()
    session.msg_logger.list_sinks.return_value = []
    session.run.side_effect = [100.0, 90.0]

    class Engine(Backtest):
        def load_messages(self, start, end):
            raise AssertionError("not used")

    engine = object.__new__(Engine)
    engine.session = session
    assert engine.get_equity() == 100.0
    assert engine.get_equity() == 90.0
    assert engine.session is session
    assert session.run.call_count == 2


def test_default_execution_submits_one_ordered_batch():
    algo = DirectExecution()
    backtest = Mock()
    backtest.time = np.datetime64("2025-01-02T09:30", "ms")
    algo.backtest = backtest
    from backtest import Direction, MarketOrder

    orders = [
        MarketOrder(
            symbol="600000.XSHG",
            time=pd.Timestamp("2025-01-01T15:00").to_pydatetime(),
            quantity=100 + i * 100,
            direction=Direction.BUY_OPEN,
        )
        for i in range(2)
    ]
    algo.on_orders(orders)
    backtest.submit_order.assert_not_called()
    backtest.submit_orders.assert_called_once()
    submitted = backtest.submit_orders.call_args.args[0]
    assert [order.quantity for order in submitted] == [100, 200]
    assert all(order.time == pd.Timestamp(backtest.time).to_pydatetime() for order in submitted)
    assert all(order.time == pd.Timestamp("2025-01-01T15:00").to_pydatetime() for order in orders)


def test_default_optimizer_retains_its_explicit_query(monkeypatch):
    from scheme.base.projects.optimize import default

    calls = []

    @contextmanager
    def query(request, **kwargs):
        calls.append((request, kwargs))
        yield SimpleNamespace(data=pd.DataFrame(columns=["time", "code", "close", "adj_factor"]))

    monkeypatch.setattr(default, "query", query)
    algo = RiskParity()
    session, settings = object(), object()
    algo.backtest = SimpleNamespace(
        time=np.datetime64("2025-01-02T15:00"), session=session, settings=settings
    )
    assert algo.allocate({"600000.XSHG": 1.0}) == {"600000.XSHG": 0.98}
    request, kwargs = calls[0]
    assert request["start_date"] == (date(2025, 1, 2) - timedelta(days=180)).isoformat()
    assert request["end_date"] == "2025-01-01"
    assert kwargs == {"session": session, "settings": settings}


def test_no_strategy_history_or_hidden_execution_dependency_remains():
    root = Path(api.__file__).resolve().parents[2]
    assert not (root / "execute/strategy/history.py").exists()
    for relative in (
        "data/dolphindb/api.py",
        "execute/strategy/api.py",
        "execute/strategy/engine.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "history_scope" not in source
        assert "current_history" not in source
        assert "CachedQueryResult" not in source
        assert "EngineSession" not in source
    assert "execute.strategy" not in (root / "data/dolphindb/api.py").read_text(encoding="utf-8")
