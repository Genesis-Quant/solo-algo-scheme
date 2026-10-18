"""离线验证进程级全时段证券域，不连接 DolphinDB。"""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, timedelta
from threading import Barrier, Event
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from runtime.apps.query import FactorQuery
from runtime.apps.query.result import QueryResult

from scheme import StockPool, Universe
from scheme.data.dolphindb import api

CODES = ("000001.SZ", "300442.SZ", "600000.SH")
CODE_QUERY = (
    "exec string(code) from (select count(*) as records from "
    f"{api._runtime.CORE_TABLE} group by code) "
    "where not isNull(code) order by code"
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(api, "_ALL_CODES", None)
    monkeypatch.setattr(
        api._runtime, "create_session", Mock(side_effect=AssertionError("禁止真实连接")),
    )
    monkeypatch.setattr(api._runtime, "redirect_session_output", Mock())
    monkeypatch.setattr(api._runtime, "has_session_variable", Mock(return_value=False))
    for name in ("execute_query", "build_query_table", "load_market_axis"):
        monkeypatch.setattr(
            api._runtime, name, Mock(side_effect=AssertionError("禁止 Runtime 重复读取代码轴")),
        )


@pytest.fixture
def make_session():
    def make():
        def run(script):
            if "exec string(code) from (select count(*)" in script:
                assert script == CODE_QUERY
                return np.asarray(CODES)
            if "exec distinct date(time)" in script:
                assert "where factor = `close" in script
                return np.asarray(["2025-01-02"], dtype="datetime64[ms]")
            return None

        return Mock(run=Mock(side_effect=run))

    return make


@pytest.fixture
def query_request():
    return {"start_date": "2025-01-02", "end_date": "2025-01-02", "codes": [], "factors": ["close"]}


@pytest.mark.parametrize("daily_first", [False, True])
def test_full_and_daily_queries_share_all_time_codes_including_missing_close(make_session, daily_first):
    core = pd.DataFrame({
        "time": pd.to_datetime(["2023-01-03", "2025-01-02", "2025-01-02", "2026-01-02"]),
        "code": ["300442.SZ", "000001.SZ", "300442.SZ", "600000.SH"],
        "factor": ["close", "close", "weight_000300SH", "close"],
    })
    daily_close = core[(core.time == pd.Timestamp("2025-01-02")) & (core.factor == "close")]
    assert "300442.SZ" not in daily_close.code.tolist()
    first, second = make_session(), make_session()
    base_run = first.run.side_effect

    def fetch(script):
        if "exec string(code) from (select count(*)" in script:
            assert script == CODE_QUERY
            return np.asarray(sorted(core.code.dropna().unique()))
        return base_run(script)

    first.run.side_effect = fetch
    intervals = [(date(2025, 1, 1), date(2025, 1, 4)), (date(2025, 1, 2), date(2025, 1, 3))]
    if daily_first:
        intervals.reverse()
    for session, (start, end) in zip((first, second), intervals, strict=True):
        original = Universe(pool=StockPool.CSI300).dataset_query(start, end, factors=["close"])
        result = api.query(original, session=session)
        assert isinstance(result, QueryResult)
        assert result.session is session
        assert original.codes == []
        uploaded = session.upload.call_args.args[0]
        assert uploaded["coreQueryCodes"].tolist() == list(CODES)
        assert uploaded["coreOutputStart"] == pd.Timestamp(start)
        assert uploaded["coreOutputEnd"] == pd.Timestamp(end)
        assert uploaded["coreDslFilters"].tolist() == []
        assert json.loads(uploaded["coreDslDefinitionsJson"]) == {
            name: node.model_dump(mode="json") for name, node in original.derivatives.items()
        }
        assert sum("exec distinct date(time)" in c.args[0] for c in session.run.call_args_list) == 1
        session.close.assert_not_called()
    assert api._ALL_CODES == CODES
    assert sum("exec string(code) from (select count(*)" in c.args[0] for c in first.run.call_args_list) == 1
    assert not any("exec string(code) from (select count(*)" in c.args[0] for c in second.run.call_args_list)
    api._runtime.create_session.assert_not_called()
    api._runtime.execute_query.assert_not_called()
    api._runtime.load_market_axis.assert_not_called()


@pytest.mark.parametrize("as_model", [False, True])
def test_explicit_codes_neither_initialize_nor_use_cache(make_session, query_request, as_model):
    query_request["codes"] = ["600000.SH", "000001.SZ"]
    original = FactorQuery.model_validate(query_request) if as_model else query_request
    expected = deepcopy(original)
    session = make_session()
    for cached in (None, CODES):
        api._ALL_CODES = cached
        api.query(original, session=session)
        uploaded = session.upload.call_args.args[0]
        assert uploaded["coreQueryCodes"].tolist() == (expected.codes if as_model else expected["codes"])
        assert api._ALL_CODES is cached
    assert original == expected
    assert not any("exec string(code) from (select count(*)" in c.args[0] for c in session.run.call_args_list)
    api._runtime.create_session.assert_not_called()


@pytest.mark.parametrize("as_model", [False, True])
def test_cached_tuple_and_caller_are_not_mutated_by_uploaded_codes(make_session, query_request, as_model):
    original = FactorQuery.model_validate(query_request) if as_model else query_request
    session = make_session()
    api.query(original, session=session)
    first = session.upload.call_args.args[0]["coreQueryCodes"]
    first[0] = "600001.SH"
    api.query(original, session=session)
    second = session.upload.call_args.args[0]["coreQueryCodes"]
    assert second.tolist() == list(CODES)
    assert second is not first
    assert api._ALL_CODES == CODES
    assert (original.codes if as_model else original["codes"]) == []
    assert sum("exec string(code) from (select count(*)" in c.args[0] for c in session.run.call_args_list) == 1


@pytest.mark.parametrize("query_request", [None, [], {}, {"start_date": "invalid"}])
def test_invalid_query_requests_fail_before_opening_or_initializing(query_request):
    with pytest.raises((TypeError, ValueError)):
        api.query(query_request)
    api._runtime.create_session.assert_not_called()
    assert api._ALL_CODES is None


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_initialization_failure_leaves_cache_retryable_and_closes_only_owned(
    monkeypatch, make_session, query_request, owned, error_type,
):
    failed = Mock(run=Mock(side_effect=error_type("fetch failed")))
    retry = make_session()
    if owned:
        monkeypatch.setattr(api._runtime, "create_session", Mock(side_effect=[failed, retry]))
    with pytest.raises(error_type, match="fetch failed"):
        api.query(query_request, **({} if owned else {"session": failed}))
    assert api._ALL_CODES is None
    assert failed.close.call_count == int(owned)
    result = api.query(query_request, **({} if owned else {"session": retry}))
    assert result.session is retry
    retry.close.assert_not_called()
    assert api._ALL_CODES == CODES
    assert retry.upload.call_args.args[0]["coreQueryCodes"].tolist() == list(CODES)
    if owned:
        assert api._runtime.create_session.call_count == 2
        api._runtime.create_session.assert_called_with(max_time=api._runtime.QUERY_SESSION_MAX_TIME)
    else:
        api._runtime.create_session.assert_not_called()


@pytest.mark.parametrize("owned", [False, True])
def test_execution_failure_closes_only_owned_session(monkeypatch, make_session, query_request, owned):
    session = make_session()
    base_run = session.run.side_effect

    def fail(script):
        if "compute_factors(" in script:
            raise RuntimeError("execute failed")
        return base_run(script)

    session.run.side_effect = fail
    if owned:
        monkeypatch.setattr(api._runtime, "create_session", Mock(return_value=session))
    with pytest.raises(RuntimeError, match="execute failed"):
        api.query(query_request, **({} if owned else {"session": session}))
    assert session.close.call_count == int(owned)
    assert api._ALL_CODES == CODES


@pytest.mark.parametrize("owned", [False, True])
def test_runtime_query_result_keeps_ownership_and_lazy_download(
    monkeypatch, make_session, query_request, owned,
):
    frame = pd.DataFrame({"code": list(CODES)})
    session = make_session()
    base_run = session.run.side_effect
    session.run.side_effect = lambda script: frame if script == api._runtime.DATA_REF else base_run(script)
    if owned:
        monkeypatch.setattr(api._runtime, "create_session", Mock(return_value=session))
    result = api.query(query_request, **({} if owned else {"session": session}))
    assert isinstance(result, QueryResult)
    assert result.session is session
    assert not result.closed
    session.close.assert_not_called()
    assert (result.source_ref, result.computed_ref, result.filtered_ref, result.data_ref) == (
        api._runtime.SOURCE_REF, api._runtime.COMPUTED_REF,
        api._runtime.FILTERED_REF, api._runtime.DATA_REF,
    )
    with pytest.raises(ValueError, match="caller failed"):
        with result as entered:
            assert entered is result
            assert result.data is frame
            raise ValueError("caller failed")
    assert result.closed
    result.close()
    session.close.assert_called_once()
    if not owned:
        api._runtime.create_session.assert_not_called()


def test_only_close_trading_dates_are_loaded_without_changing_lookback(make_session):
    trading_dates = np.asarray(["2025-01-03", "2025-01-06"], dtype="datetime64[ms]")
    session = make_session()
    base_run = session.run.side_effect
    session.run.side_effect = lambda script: (
        trading_dates if "exec distinct date(time)" in script else base_run(script)
    )
    request = Universe().dataset_query(
        date(2025, 1, 3), date(2025, 1, 7), factors=["close"], lookback=timedelta(days=1),
    )
    with api.query(request, session=session):
        uploaded = session.upload.call_args.args[0]
        assert uploaded["coreQueryCodes"].tolist() == list(CODES)
        np.testing.assert_array_equal(uploaded["coreQueryDates"], trading_dates)
        assert uploaded["coreQueryStart"] == pd.Timestamp("2025-01-02")
        assert uploaded["coreQueryEnd"] == pd.Timestamp("2025-01-07")
        assert uploaded["coreOutputStart"] == pd.Timestamp("2025-01-03")
    assert request.codes == []


def test_source_cache_keeps_existing_runtime_contract(monkeypatch, make_session, query_request):
    monkeypatch.setattr(api._runtime, "has_session_variable", Mock(return_value=True))
    session = make_session()
    with api.query(query_request, session=session):
        scripts = "\n".join(c.args[0] for c in session.run.call_args_list)
        assert "build_factor_source(" not in scripts
        assert "finalize_factor_source(" not in scripts
        assert "compute_factors(" in scripts
        assert "filter_factors(" in scripts
        assert "project_factor_output(" in scripts
    api._runtime.has_session_variable.assert_called_once_with(session, api._runtime.SOURCE_REF)


def test_query_preserves_runtime_fills_and_only_loads_query_module(monkeypatch, make_session):
    financial = next(iter(api._runtime.FINANCIAL_FACTORS))
    industry = next(iter(api._runtime.INDUSTRY_FACTORS))
    factors = ["close", api._runtime.IS_ST_FACTOR, "weight_000300SH", financial, industry]
    monkeypatch.setattr(api._runtime, "available_factors", lambda: factors)
    session = make_session()
    with api.query({
        "start_date": "2025-01-02", "end_date": "2025-01-02", "codes": [], "factors": factors,
    }, session=session):
        scripts = [c.args[0] for c in session.run.call_args_list]
        assert [s for s in scripts if s.startswith("use ")] == ["use query"]
        fills = "\n".join(scripts)
        assert "fill_null_column(" in fills
        assert "fill_observed_group_null_column(" in fills
        assert fills.count("forward_fill_column(") == 3
        assert fills.index("fill_observed_group_null_column(") < fills.index("forward_fill_column(")
        for name in (financial, industry):
            assert json.dumps(name) in fills
        assert session.upload.call_args.args[0]["coreQuerySeedFactors"].tolist() == [industry]


@pytest.mark.parametrize("owned", [False, True])
def test_missing_close_dates_fail_without_synthetic_calendar(monkeypatch, make_session, query_request, owned):
    session = make_session()
    base_run = session.run.side_effect
    session.run.side_effect = lambda script: (
        np.asarray([], dtype="datetime64[ms]")
        if "exec distinct date(time)" in script else base_run(script)
    )
    if owned:
        monkeypatch.setattr(api._runtime, "create_session", Mock(return_value=session))
    with pytest.raises(ValueError, match="没有收盘价数据"):
        api.query(query_request, **({} if owned else {"session": session}))
    assert session.close.call_count == int(owned)
    assert session.upload.call_count == 1


def test_concurrent_first_empty_queries_fetch_codes_once(make_session, query_request):
    start = Barrier(3)
    fetching, release = Event(), Event()

    def fetch():
        fetching.set()
        assert release.wait(timeout=5)
        return np.asarray(CODES)

    fetch = Mock(side_effect=fetch)

    def run():
        session = make_session()
        base_run = session.run.side_effect
        session.run.side_effect = lambda script: fetch() if script == CODE_QUERY else base_run(script)
        start.wait(timeout=5)
        return api.query(query_request, session=session)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(run), pool.submit(run)
        start.wait(timeout=5)
        try:
            assert fetching.wait(timeout=5)
        finally:
            release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    fetch.assert_called_once()
    for result in results:
        assert result.session.upload.call_args.args[0]["coreQueryCodes"].tolist() == list(CODES)
