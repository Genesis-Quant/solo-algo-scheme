"""History and its template consumers stay entirely in memory in these tests."""

from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from runtime.apps.query import FactorQuery

import scheme.data
from scheme.base.projects.optimize import default as optimize
from scheme.data.dolphindb.symbols import engine_symbol
from scheme.execute.strategy.history import History, current_history, history_scope


@pytest.fixture
def prepared(monkeypatch):
    dates = pd.bdate_range("2024-06-01", "2025-08-15")
    rows = []
    for i, code in enumerate(["600000.SH", "000001.SZ", "000002.SZ"]):
        for j, day in enumerate(dates):
            rows.append((day, code, 20 * np.exp(j * 0.0003 * (i + 1)) + np.sin(j / (i + 2)), 1.0))
    data = pd.DataFrame(rows, columns=["time", "code", "close", "adj_factor"])
    data.loc[(data.code == "000002.SZ") & (data.time == "2025-07-01"), "close"] = np.nan
    data.loc[(data.code == "000001.SZ") & (data.time == "2025-07-03"), "adj_factor"] = np.nan
    now = [pd.NaT]
    session, settings = object(), object()
    calls = []

    @contextmanager
    def query(request, **kwargs):
        assert current_history() is None
        assert kwargs == {"session": session, "settings": settings}
        calls.append(request)
        mask = data.time.between(request["start_date"], request["end_date"])
        mask &= data.code.isin(request["codes"])
        calendar = dates[(dates >= request["start_date"]) & (dates <= request["end_date"])]
        yield SimpleNamespace(data=data.loc[mask].copy(), calendar=calendar)

    monkeypatch.setattr(scheme.data, "query", query)
    history = History(
        session,
        settings,
        "2025-01-01",
        "2025-08-01",
        ["600000.XSHG", "000001.SZ", "000002.XSHE"],
        lookback_days=180,
        clock=lambda: now[0],
    )
    with history_scope(history):
        history.prepare()
        assert current_history() is history
    now[0] = pd.Timestamp("2025-07-15 15:00")
    return SimpleNamespace(
        history=history,
        now=now,
        data=data,
        calls=calls,
        session=session,
        settings=settings,
        dates=dates,
    )


def legacy_frame(prepared, start, end, codes, fields=("close", "adj_factor")):
    data = prepared.data
    return data.loc[
        data.time.between(pd.Timestamp(start), pd.Timestamp(end)) & data.code.isin(codes),
        ["time", "code", *fields],
    ].reset_index(drop=True)


def request(**changes):
    values = {
        "start_date": "2025-05-16",
        "end_date": "2025-07-14",
        "codes": ["600000.SH"],
        "factors": ["close", "adj_factor"],
    }
    values.update(changes)
    return FactorQuery.model_validate(values)


def test_prepare_is_one_query_and_bypasses_nat(prepared):
    prepared.now[0] = pd.NaT
    prepared.history.prepare()
    assert prepared.calls == [
        {
            "start_date": "2024-07-05",
            "end_date": "2025-07-31",
            "codes": ["600000.SH", "000001.SZ", "000002.SZ"],
            "factors": ["close", "adj_factor"],
        }
    ]
    expected = prepared.dates[(prepared.dates >= "2024-07-05") & (prepared.dates < "2025-08-01")]
    assert prepared.history.calendar.equals(expected)


def test_explicit_window_is_exact_copy_with_source_codes(prepared):
    actual = prepared.history.query("2025-05-16", "2025-07-14", ["600000.XSHG", "000002.XSHE"])
    expected = (
        legacy_frame(prepared, "2025-05-16", "2025-07-14", ["000002.SZ", "600000.SH"])
        .sort_values(["code", "time"])
        .reset_index(drop=True)
    )
    assert_frame_equal(actual, expected)
    actual.loc[:, "close"] = -1
    again = prepared.history.query("2025-05-16", "2025-07-14", ["600000.SH", "000002.SZ"])
    assert_frame_equal(again, expected)
    assert len(prepared.calls) == 1


def test_empty_codes_does_not_mean_market(prepared):
    actual = prepared.history.query("2025-07-01", "2025-07-14", [])
    assert actual.empty
    assert actual.columns.tolist() == ["time", "code", "close", "adj_factor"]
    assert len(prepared.calls) == 1


@pytest.mark.parametrize("clock", [pd.NaT, np.datetime64("NaT", "ms"), None])
def test_nat_is_not_a_readable_simulation_clock(prepared, clock):
    prepared.now[0] = clock
    with pytest.raises(ValueError, match="NaT"):
        prepared.history.query("2025-07-01", "2025-07-14", ["600000.SH"])
    assert prepared.history.get(request()) is None


@pytest.mark.parametrize("clock", ["2025-07-15 09:30", "2025-07-15 14:59:59"])
def test_today_is_unavailable_before_close_even_when_opted_in(prepared, clock):
    prepared.now[0] = pd.Timestamp(clock)
    with pytest.raises(ValueError, match="15:00"):
        prepared.history.query("2025-07-01", "2025-07-15", ["600000.SH"], include_today=True)
    with pytest.raises(ValueError, match="15:00"):
        prepared.history.get(request(end_date="2025-07-15"))


def test_today_requires_explicit_opt_in_after_close(prepared):
    with pytest.raises(ValueError, match="include_today"):
        prepared.history.query("2025-07-01", "2025-07-15", ["600000.SH"])
    result = prepared.history.query("2025-07-01", "2025-07-15", ["600000.SH"], include_today=True)
    assert result.time.max() == pd.Timestamp("2025-07-15")
    assert_frame_equal(
        prepared.history.get(request(start_date="2025-07-01", end_date="2025-07-15")), result
    )


def test_future_is_never_a_cache_miss(prepared):
    with pytest.raises(ValueError, match="模拟时点"):
        prepared.history.query("2025-07-01", "2025-08-10", ["600000.SH"], include_today=True)
    with pytest.raises(ValueError, match="模拟时点"):
        prepared.history.get(request(end_date="2025-08-10", factors=["open"]))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"start_date": "2024-07-04"}, "日期范围"),
        ({"codes": ["600001.SH"]}, "未预加载"),
        ({"fields": ["open"]}, "close 和 adj_factor"),
        ({"start_date": "2025-07-15"}, "不能晚于"),
    ],
)
def test_explicit_query_does_not_expand_or_fallback(prepared, changes, message):
    values = {
        "start_date": "2025-07-01",
        "end_date": "2025-07-14",
        "codes": ["600000.SH"],
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        prepared.history.query(**values)
    assert len(prepared.calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"codes": []},
        {"codes": ["600001.SH"]},
        {"factors": ["open"]},
        {"lookback": timedelta(days=1)},
        {"start_date": "2024-07-04"},
        {
            "derivatives": {
                "ret": {
                    "type": "TS",
                    "op": "unary.pct_change",
                    "fields": {"col": "close"},
                    "params": {"periods": 1},
                }
            }
        },
    ],
)
def test_only_raw_prices_whitelist_is_cached(prepared, changes):
    assert prepared.history.get(request(**changes)) is None


def test_whitelist_field_projection_is_exact(prepared):
    result = prepared.history.get(request(factors=["adj_factor", "close"]))
    expected = legacy_frame(
        prepared, "2025-05-16", "2025-07-14", ["600000.SH"], ["adj_factor", "close"]
    )
    assert_frame_equal(result, expected)


def test_scope_restores_nested_context_after_exception(prepared):
    assert current_history() is None
    with history_scope(prepared.history):
        assert current_history() is prepared.history
        with pytest.raises(RuntimeError), history_scope(None):
            assert current_history() is None
            raise RuntimeError("intentional")
        assert current_history() is prepared.history
    assert current_history() is None


@pytest.mark.parametrize("fields", [("close", "adj_factor"), ("adj_factor",), ()])
def test_indexed_windows_match_legacy_rows_exactly(prepared, fields):
    rng = np.random.default_rng(42)
    codes = np.array(prepared.history.codes)
    for _ in range(40):
        end = pd.Timestamp("2025-07-14") - timedelta(days=int(rng.integers(0, 120)))
        start = end - timedelta(days=int(rng.integers(0, 180)))
        count = int(rng.integers(0, len(codes) + 1))
        selected = rng.choice(codes, count, replace=False).tolist()
        actual = prepared.history.query(start, end, list(reversed(selected)), fields=fields)
        expected = (
            legacy_frame(prepared, start, end, selected, fields)
            .sort_values(["code", "time"])
            .reset_index(drop=True)
        )
        assert_frame_equal(actual, expected, check_exact=True)


def test_normalization_cache_is_bounded_and_reused_by_get(prepared, monkeypatch):
    from scheme.execute.strategy import history as module

    normalize = Mock(wraps=module.source_symbol)
    monkeypatch.setattr(module, "source_symbol", normalize)
    values = request(codes=["600000.SH", "000002.SZ"])
    first = prepared.history.get(values)
    second = prepared.history.get(values)
    assert normalize.call_count == 2
    assert_frame_equal(first, second, check_exact=True)
    for count in range(40):
        prepared.history.query("2025-07-01", "2025-07-14", ["600000.SH" + " " * count])
    assert prepared.history._cached_codes.cache_info().currsize <= 32
    other = History(
        object(),
        object(),
        "2025-01-01",
        "2025-08-01",
        ["600000.SH", "000002.SZ"],
        clock=lambda: prepared.now[0],
    )
    assert other._cached_codes.cache_info().currsize == 1
    assert normalize.call_count == 44


def test_empty_preload_keeps_server_calendar(monkeypatch):
    calendar = pd.bdate_range("2024-11-02", "2025-01-31")

    @contextmanager
    def query(*args, **kwargs):
        yield SimpleNamespace(
            data=pd.DataFrame(columns=["time", "code", "close", "adj_factor"]),
            calendar=calendar,
        )

    monkeypatch.setattr(scheme.data, "query", query)
    history = History(
        object(),
        object(),
        "2025-01-01",
        "2025-02-01",
        ["600000.SH"],
        lookback_days=1,
        clock=lambda: pd.Timestamp("2025-01-15 15:00"),
    )
    history.prepare()
    assert history.lookback_days == 60
    assert history.calendar.equals(calendar)
    assert history.query("2025-01-01", "2025-01-14", ["600000.SH"]).empty


def test_empty_preload_codes_never_query_database(monkeypatch):
    query = Mock(side_effect=AssertionError("must not query whole market"))
    monkeypatch.setattr(scheme.data, "query", query)
    history = History(
        object(),
        object(),
        "2025-01-01",
        "2025-02-01",
        [],
        clock=lambda: pd.Timestamp("2025-01-15 15:00"),
    )
    history.prepare()
    assert history.query("2025-01-01", "2025-01-14", []).empty
    query.assert_not_called()


def test_risk_parity_matches_legacy_window_and_signal_order(prepared, monkeypatch):
    symbols = ["600000.XSHG", "000002.XSHE", "000001.XSHE"]
    signals = {symbols[0]: 1.0, symbols[1]: -1.0, symbols[2]: 1.0}
    baseline = legacy_frame(prepared, "2025-01-16", "2025-07-14", list(prepared.history.codes))
    baseline["code"] = baseline.code.map(engine_symbol)
    baseline["adjusted"] = baseline.close * baseline.adj_factor
    prices = baseline.pivot(index="time", columns="code", values="adjusted").reindex(
        columns=symbols
    )
    expected_returns = prices.pct_change(fill_method=None).mul([1, -1, 1])
    expected_weights, reason = optimize.risk_parity(expected_returns)
    assert reason is None
    assert expected_returns.iloc[0].isna().all()
    assert expected_returns.isna().any().all()
    compute = optimize.risk_parity
    captured = []

    def capture(returns):
        captured.append(returns.copy())
        return compute(returns)

    monkeypatch.setattr(optimize, "risk_parity", capture)
    fallback = Mock(side_effect=AssertionError("history path must not query"))
    monkeypatch.setattr(optimize, "query", fallback)
    algo = optimize.RiskParity()
    algo.backtest = SimpleNamespace(history=prepared.history, time=prepared.now[0])
    actual = algo.allocate(signals)
    assert list(actual) == symbols
    assert_frame_equal(captured[0], expected_returns)
    np.testing.assert_allclose(
        list(actual.values()), expected_weights * np.array([1, -1, 1]) * 0.98
    )
    assert algo.diagnostics[0]["reason"] is None
    fallback.assert_not_called()


def test_risk_parity_fallback_uses_backtest_session(prepared, monkeypatch):
    calls = []

    @contextmanager
    def query(values, **kwargs):
        calls.append((values, kwargs))
        yield SimpleNamespace(
            data=legacy_frame(prepared, values["start_date"], values["end_date"], values["codes"])
        )

    monkeypatch.setattr(optimize, "query", query)
    algo = optimize.RiskParity()
    algo.backtest = SimpleNamespace(
        time=prepared.now[0], session=prepared.session, settings=prepared.settings
    )
    algo.allocate({"600000.XSHG": 1.0})
    values, kwargs = calls[0]
    assert values["start_date"] == "2025-01-16"
    assert values["end_date"] == "2025-07-14"
    assert kwargs == {"session": prepared.session, "settings": prepared.settings}


@pytest.mark.parametrize("use_history", [False, True])
def test_model_matches_legacy_scores_and_daily_membership(prepared, monkeypatch, use_history):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "model" / "src"))
    from model import algo as model
    from model.params import ModelParams

    codes = ["600000.SH", "000001.SZ"]
    expected = legacy_frame(prepared, "2025-05-16", "2025-07-15", codes)
    expected["price"] = expected.close * expected.adj_factor
    prices = expected.pivot(index="time", columns="code", values="price").sort_index()
    scores = prices.pct_change(20, fill_method=None).iloc[-1].dropna()
    expected_signal = {symbol: 1.0 for symbol in scores.nlargest(2).index}
    calls = []

    @contextmanager
    def query(values, **kwargs):
        assert not use_history
        calls.append((values, kwargs))
        yield SimpleNamespace(data=expected[["time", "code", "close", "adj_factor"]])

    monkeypatch.setattr(model, "query", query)
    universe = pd.DataFrame(
        [[True, True, False]],
        index=[pd.Timestamp("2025-07-15")],
        columns=["600000.XSHG", "000001.XSHE", "000002.XSHE"],
    )
    algo = model.ModelAlgo(ModelParams(n_select=2))
    algo.backtest = SimpleNamespace(
        time=pd.Timestamp("2025-07-15 09:30"),
        universe=universe,
        session=prepared.session,
        settings=prepared.settings,
    )
    if use_history:
        algo.backtest.history = prepared.history
    algo.ctx = SimpleNamespace(signal=None)
    algo.initialize()
    algo.on_snapshot(None)
    assert algo.ctx.signal is None
    algo.backtest.time = prepared.now[0]
    algo.on_snapshot(None)
    assert algo.ctx.signal == expected_signal
    assert list(algo.ctx.signal) == list(expected_signal)
    if not use_history:
        values, kwargs = calls[0]
        assert values["start_date"] == "2025-05-16"
        assert values["end_date"] == "2025-07-15"
        assert values["codes"] == codes
        assert kwargs == {"session": prepared.session, "settings": prepared.settings}
    assert len(prepared.calls) == 1
