from unittest.mock import Mock

import pytest

from scheme.execute.strategy.engine import EngineSession


def test_session_call_flushes_pending_events_before_invalidating_snapshot():
    owner = Mock()
    session = Mock()
    calls = []
    owner.flush_pending_events.side_effect = lambda **kwargs: calls.append(
        ("flush", kwargs["session"])
    )
    owner.invalidate_snapshot.side_effect = lambda: calls.append(("invalidate", None))
    session.run.side_effect = lambda *args, **kwargs: calls.append(("run", args)) or 42
    proxy = EngineSession(session, owner)
    assert proxy.run("query") == 42
    assert calls == [("flush", session), ("invalidate", None), ("run", ("query",))]


def test_session_upload_flushes_before_writing():
    owner = Mock()
    session = Mock()
    proxy = EngineSession(session, owner)
    proxy.upload({"value": 1})
    owner.flush_pending_events.assert_called_once_with(session=session)
    owner.invalidate_snapshot.assert_called_once_with()
    session.upload.assert_called_once_with({"value": 1})


def test_other_session_execution_entry_points_cannot_bypass_invalidation():
    owner = Mock()
    session = Mock()
    session.exec.return_value = 42
    proxy = EngineSession(session, owner)
    assert proxy.exec("query") == 42
    owner.flush_pending_events.assert_called_once_with(session=session)
    owner.invalidate_snapshot.assert_called_once_with()


def test_pending_event_confirmation_failure_prevents_raw_execution():
    owner = Mock()
    owner.flush_pending_events.side_effect = RuntimeError("event prefix changed")
    session = Mock()
    proxy = EngineSession(session, owner)
    with pytest.raises(RuntimeError, match="event prefix changed"):
        proxy.run("write")
    session.run.assert_not_called()
    owner.invalidate_snapshot.assert_not_called()
