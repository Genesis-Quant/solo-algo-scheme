from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scheme.execute.strategy.api import ResearchBacktest


def adapter(times, counts):
    engine = object.__new__(ResearchBacktest)
    engine.parameters = SimpleNamespace(market_data="stock_daily")
    engine._daily_timeline = np.asarray(times, dtype="datetime64[ms]")
    engine._daily_counts = np.asarray(counts, dtype=np.int64)
    engine._loaded_daily_batches = set()
    engine.loaded_rows = 0
    engine.loaded_dates = set()
    engine.session = Mock()
    return engine


def test_market_batches_use_only_their_own_time_range_and_statistics():
    engine = adapter(
        ["2025-01-02T09:30", "2025-01-02T15:00", "2025-01-03T09:30", "2025-01-03T15:00"],
        [2, 2, 3, 3],
    )
    start = np.datetime64("2025-01-02", "ms")
    end = np.datetime64("2025-01-03", "ms")
    messages = engine.load_messages(start, end)
    assert str(messages) == "soloMessages"
    sql = engine.session.run.call_args.args[0]
    assert "select * from soloDailyMessages" in sql
    assert f"timestamp >= timestamp({start.astype('int64')}l)" in sql
    assert f"timestamp < timestamp({end.astype('int64')}l)" in sql
    np.testing.assert_array_equal(
        engine.load_timeline(messages, start, end), engine._daily_timeline[:2]
    )
    assert engine.loaded_rows == 4
    assert engine.loaded_dates == {date(2025, 1, 2)}
    engine.load_messages(start, end)
    assert engine.loaded_rows == 4
    engine.load_messages(end, end + np.timedelta64(1, "D"))
    assert engine.loaded_rows == 10
    assert engine.loaded_dates == {date(2025, 1, 2), date(2025, 1, 3)}


def test_empty_batch_keeps_empty_timeline_without_research_data():
    engine = adapter(["2025-01-02T09:30"], [2])
    start = np.datetime64("2025-01-04", "ms")
    end = start + np.timedelta64(1, "D")
    message = engine.load_messages(start, end)
    assert len(engine.load_timeline(message, start, end)) == 0
    assert engine.loaded_rows == 0
    assert not engine.loaded_dates
    assert engine.session.run.call_count == 1


def test_empty_market_symbols_never_query_whole_market(monkeypatch):
    from scheme.execute.strategy import api

    engine = adapter([], [])
    engine.parameters = SimpleNamespace(
        market_data="stock_daily", start=date(2025, 1, 1), end=date(2025, 1, 10)
    )
    engine.symbols = []
    engine._daily_timeline = None
    monkeypatch.setattr(api, "query", Mock(side_effect=AssertionError("no whole-market query")))
    engine._prepare_daily_messages()
    assert engine._daily_timeline.size == 0
    assert engine._daily_counts.size == 0
    assert "soloDailyMessages" in engine.session.run.call_args.args[0]


@pytest.mark.parametrize("batch_days", [1, 2, 5])
def test_local_timeline_preserves_each_time_once_across_batches(batch_days):
    times = pd.date_range("2025-01-02T09:30", periods=8, freq="D").to_numpy(dtype="datetime64[ms]")
    engine = adapter(times, [2] * len(times))
    start = np.datetime64("2025-01-01", "ms")
    end = np.datetime64("2025-01-11", "ms")
    observed = []
    while start < end:
        stop = min(start + np.timedelta64(batch_days, "D"), end)
        message = engine.load_messages(start, stop)
        observed.extend(engine.load_timeline(message, start, stop))
        start = stop
    np.testing.assert_array_equal(observed, times)
    assert engine.loaded_rows == 16
