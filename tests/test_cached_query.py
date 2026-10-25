from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scheme.data.dolphindb.api import CachedQueryResult, query
from scheme.execute.strategy.history import history_scope


def frame():
    return pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "code": ["600000.SH", "600000.SH"],
            "close": [10.0, 11.0],
        }
    )


def test_cached_query_has_no_network_without_server_reference():
    session = Mock()
    result = CachedQueryResult(frame(), pd.to_datetime(["2025-01-02", "2025-01-03"]), session)
    with result:
        data = result.data
        data.loc[0, "close"] = -1
        assert result.data.close.tolist() == [10.0, 11.0]
        assert len(result.calendar) == 2
    session.upload.assert_not_called()
    session.run.assert_not_called()
    with pytest.raises(RuntimeError, match="已关闭"):
        _ = result.data


def test_cached_query_materializes_only_its_own_server_variables():
    session = Mock()
    session.msg_logger.list_sinks.return_value = []
    result = CachedQueryResult(frame(), pd.to_datetime(["2025-01-02", "2025-01-03"]), session)
    data_ref, dates_ref = result.data_ref, result.calendar_ref
    assert data_ref.startswith("historyData_")
    assert dates_ref.startswith("historyDates_")
    assert set(session.upload.call_args.args[0]) == {data_ref, dates_ref}
    assert session.upload.call_count == 1
    session.run.return_value = frame()
    pd.testing.assert_frame_equal(result.data, frame())
    session.run.return_value = np.array(["2025-01-03"], dtype="datetime64[ms]")
    np.testing.assert_array_equal(result.calendar, session.run.return_value)
    assert session.run.call_args.args[0] == dates_ref
    result.close()
    script = session.run.call_args.args[0]
    assert data_ref in script and dates_ref in script
    assert '"queryData"' not in script
    session.close.assert_not_called()


def test_cached_calendar_retains_days_without_selected_records():
    result = CachedQueryResult(
        frame().iloc[:0], pd.to_datetime(["2025-01-02", "2025-01-03"]), Mock()
    )
    assert result.data.empty
    np.testing.assert_array_equal(
        result.calendar, np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[ms]")
    )


def test_original_query_uses_active_history_without_new_connection(monkeypatch):
    session = Mock()
    history = Mock(
        session=session,
        calendar=pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    history.get.return_value = frame()
    monkeypatch.setattr(
        "scheme.data.dolphindb.api.open_session", Mock(side_effect=AssertionError("must reuse"))
    )
    with history_scope(history):
        with query(
            {
                "start_date": "2025-01-01",
                "end_date": "2025-01-03",
                "codes": ["600000.SH"],
                "factors": ["close"],
            }
        ) as result:
            pd.testing.assert_frame_equal(result.data, frame())
            assert len(result.calendar) == 2
    session.upload.assert_not_called()
    session.run.assert_not_called()
    assert history.get.call_args.args[0].lookback == timedelta(0)
