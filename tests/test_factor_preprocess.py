"""Offline transport/source contracts; no Python approximation of DOS regression."""

import re
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from backtest import DosVar
from runtime.apps.query import FactorQuery

from scheme import Factor, FactorParams
from scheme.config import DolphinSettings

module = import_module("scheme.base.projects.factor.preprocess")
SETTINGS = DolphinSettings(host="unused.invalid", username="offline", password="unused")


def raw_frame():
    return pd.DataFrame({
        "time": pd.to_datetime(["2025-01-06", "2025-01-03", "2025-01-03"]),
        "code": ["000001.SZ", "600000.SH", "000001.SZ"],
        "alpha": [7.0, 2.0, np.nan], "beta": [3.0, np.nan, 5.0],
    })


def raw_exposures():
    return raw_frame()[["time", "code"]].assign(
        circ_mv=[1.0, 4.0, 6.0], industry_l0=["bank", "bank", None],
    )


class FakeSession:
    """Only transports fixed server responses; deliberately does not neutralize."""

    def __init__(self, source, response=None):
        self.source = source.copy(deep=True)
        self.response = response.copy(deep=True) if response is not None else source.copy(deep=True)
        self.scripts = []
        self.process_scripts = []
        self.uploads = []
        self.variables = {"coreQueryData": self.source, "coreQuerySourceData": "caller cache"}
        self.closed = False
        self.fail_process = False
        self.fail_cleanup = False
        self.fail_upload = False

    def upload(self, values):
        self.uploads.append(values)
        self.variables.update(values)
        if self.fail_upload:
            raise RuntimeError("upload failed")

    def run(self, script):
        self.scripts.append(script)
        if re.search(r"\bisDef\s*\(", script):
            raise RuntimeError("DolphinDB does not support isDef")
        if script.startswith("columnNames("):
            return np.asarray(self.source.columns)
        if script.startswith("schema("):
            return pd.DataFrame({
                "name": self.source.columns,
                "typeString": [
                    "BOOL" if pd.api.types.is_bool_dtype(self.source[column])
                    else "COMPLEX" if pd.api.types.is_complex_dtype(self.source[column])
                    else "DOUBLE" for column in self.source.columns
                ],
            })
        if script.startswith("select time, code from "):
            return self.source[["time", "code"]].copy()
        if "undef(" in script:
            if self.fail_cleanup:
                raise RuntimeError("cleanup failed")
            statements = script.split("; ")
            pattern = r'if \("([A-Za-z0-9_]+)" in objs\(\)\.name\) \{ undef\("\1", VAR\) \}'
            for statement in statements:
                match = re.fullmatch(pattern, statement)
                if match is None:
                    raise AssertionError(f"Unsupported cleanup statement: {statement}")
                self.variables.pop(match.group(1), None)
            return None
        if re.match(r"\nsoloFp[0-9a-f]{32}Working = ", script):
            self.process_scripts.append(script)
            assert not re.search(r"\bdef\b", script)
            assert "soloFactorPreprocess" not in script
            # Track private temporaries, not DOS values or numerical execution.
            for name in re.findall(r"\bsoloFp[0-9a-f]{32}[A-Za-z]+\b", script):
                self.variables[name] = "server temporary"
            if self.fail_process:
                raise RuntimeError("DOS failed")
            return self.response.copy(deep=True)
        return None

    def close(self):
        self.closed = True


@pytest.fixture
def sessions(monkeypatch):
    owned = []

    @contextmanager
    def create(settings):
        assert settings is SETTINGS
        session = FakeSession(raw_frame())
        owned.append(session)
        try:
            yield session
        finally:
            session.close()

    def query(request, *, session):
        FactorQuery.model_validate(request)
        assert session in owned and not session.closed
        return SimpleNamespace(data=raw_exposures())

    monkeypatch.setattr(module, "create_session", create)
    monkeypatch.setattr(module, "query", query)
    # Prevent any offline test from falling through to environment configuration.
    monkeypatch.setattr(DolphinSettings, "from_env", classmethod(lambda cls: SETTINGS))
    return owned


def expected_panel(frame, columns):
    panel = frame[["time", "code", *columns]].copy()
    panel["code"] = panel["code"].map(module.engine_symbol)
    panel = panel.rename(columns={"time": "date", "code": "symbol"})
    return panel.set_index(["date", "symbol"])[columns].astype(float).sort_index()


def test_dataframe_uses_dos_and_returns_only_selected_sorted_canonical_columns(sessions):
    data = raw_frame()
    data["stock_pool_member"] = True
    data["unselected"] = ["not", "numeric", "extra"]
    original = data.copy(deep=True)
    # Fixed server NULLs represent invalid exposures/degenerate sections; keep every key.
    result_frame = data.copy()
    result_frame["alpha"] = [np.nan, 1.0, np.nan]
    result_frame["beta"] = [np.nan, np.nan, -1.0]
    session = FakeSession(data, result_frame)
    result = module.preprocess(data, session=session, columns=["beta", "alpha"])
    pd.testing.assert_frame_equal(result, expected_panel(result_frame, ["beta", "alpha"]))
    pd.testing.assert_frame_equal(data, original)
    assert result.index.is_unique and result.index.is_monotonic_increasing
    assert len(result) == len(data) and result.isna().sum().sum() == 4
    uploaded = session.uploads[0]
    input_frame = next(value for key, value in uploaded.items() if key.endswith("Input"))
    assert list(input_frame) == ["time", "code", "beta", "alpha"]
    assert all(re.fullmatch(r"soloFp[0-9a-f]{32}[A-Za-z]+", name) for name in uploaded)
    assert len(session.process_scripts) == 1
    assert len(session.scripts) == 2  # One inline calculation and its finally cleanup.
    assert all(not re.search(r"\bdef\b", script) for script in session.scripts)
    assert all("soloFactorPreprocess" not in script for script in session.scripts)
    assert session.variables == {"coreQueryData": session.source, "coreQuerySourceData": "caller cache"}
    assert not session.closed and len(sessions) == 1 and sessions[0].closed


def test_local_dataframe_owned_processing_and_exposure_sessions_are_distinct_and_closed(sessions):
    result = module.preprocess(raw_frame(), columns=["alpha", "beta"], settings=SETTINGS)
    assert list(result) == ["alpha", "beta"]
    assert len(sessions) == 2 and all(session.closed for session in sessions)
    assert sessions[0] is not sessions[1]
    assert len(sessions[0].process_scripts) == 1 and not sessions[1].process_scripts
    assert not any(name.startswith("soloFp") for name in sessions[0].variables)


def test_from_env_is_only_used_for_owned_sessions_and_does_not_load_dotenv(sessions):
    module.preprocess(raw_frame(), columns=["alpha", "beta"])
    assert len(sessions) == 2 and all(session.closed for session in sessions)


def test_canonical_input_is_supported_without_modification(sessions):
    canonical = expected_panel(raw_frame(), ["alpha", "beta"])
    canonical["circ_mv"] = [4.0, 6.0, 1.0]
    canonical["industry_l0"] = ["bank", "bank", "bank"]
    original = canonical.copy(deep=True)
    session = FakeSession(raw_frame())
    result = module.preprocess(canonical, session=session, columns=["alpha", "beta"])
    assert list(result.index.names) == ["date", "symbol"]
    pd.testing.assert_frame_equal(canonical, original)
    uploaded = next(value for name, value in session.uploads[0].items() if name.endswith("Input"))
    assert set(uploaded.code) == {"000001.SZ", "600000.SH"}


def test_dosvar_borrows_session_downloads_only_keys_and_never_mutates_original(sessions):
    source = raw_frame()
    source["stock_pool_member"] = True
    source["intermediate"] = "unused query node"
    source["circ_mv"] = -999.0
    source["industry_l0"] = "ignore caller exposure"
    original = source.copy(deep=True)
    session = FakeSession(source)
    result = module.preprocess(DosVar("coreQueryData"), session=session, columns=["alpha", "beta"])
    pd.testing.assert_frame_equal(result, expected_panel(source, ["alpha", "beta"]))
    pd.testing.assert_frame_equal(session.source, original)
    assert not session.closed and len(sessions) == 1 and sessions[0].closed
    assert "select time, code from coreQueryData" in session.scripts
    prefix = re.search(r"soloFp[0-9a-f]{32}", session.process_scripts[0]).group()
    assert f'Working = sql(sqlCol(symbol(["time", "code"] join {prefix}Factors)), coreQueryData).eval()' in session.process_scripts[0]
    assert "Working = coreQueryData.copy()" not in session.process_scripts[0]
    assert "intermediate" not in session.process_scripts[0]
    assert "stock_pool_member" not in session.process_scripts[0]
    assert len(session.process_scripts) == 1
    assert all(not key.endswith("Input") for key in session.uploads[0])
    cleanup = session.scripts[-1]
    assert "coreQueryData" not in cleanup and "coreQuerySourceData" not in cleanup
    assert not any("use factor" in script or "factor::" in script for script in session.scripts)


@pytest.mark.parametrize("use_variable", [False, True])
@pytest.mark.parametrize("caller_exposures", [False, True])
def test_exposures_always_query_exact_input_range_codes_in_separate_owned_session(
    monkeypatch, sessions, use_variable, caller_exposures,
):
    source = raw_frame()
    if caller_exposures:
        source["circ_mv"] = [np.nan, -999.0, 0.0]
        source["industry_l0"] = "must not be reused"
    source["stock_pool_member"] = False
    source["intermediate"] = "not a factor"
    caller = FakeSession(source)
    calls = []
    queried = raw_exposures()

    def query(request, *, session):
        validated = FactorQuery.model_validate(request)
        assert validated.lookback.total_seconds() == 0
        assert session is not caller and session in sessions
        assert not session.closed
        assert "coreQuerySourceData" not in request
        calls.append((request, session))
        # Extra dates/codes from a provider cannot expand the output keys.
        exposure = pd.concat([queried, pd.DataFrame({
            "time": pd.to_datetime(["2025-01-02", "2025-01-07"]),
            "code": ["999999.SZ", "000001.SZ"], "circ_mv": [8.0, 9.0],
            "industry_l0": ["other", "future"],
        })], ignore_index=True).sample(frac=1, random_state=42)
        return SimpleNamespace(data=exposure)

    monkeypatch.setattr(module, "query", query)
    result = module.preprocess(
        DosVar("coreQueryData") if use_variable else source,
        session=caller, columns=["alpha", "beta"], settings=SETTINGS,
    )
    assert len(result) == len(source)
    request, owned = calls[0]
    assert request == {
        "start_date": "2025-01-03", "end_date": "2025-01-06",
        "codes": ["000001.SZ", "600000.SH"], "factors": ["circ_mv", "industry_l0"],
    }
    assert len(calls) == 1 and owned.closed and not caller.closed
    uploaded = next(value for name, value in caller.uploads[0].items() if name.endswith("Exposure"))
    pd.testing.assert_frame_equal(uploaded, queried, check_dtype=False)
    assert any(" = lj(" in script for script in caller.scripts)
    if not use_variable:
        uploaded_input = next(value for name, value in caller.uploads[0].items() if name.endswith("Input"))
        assert list(uploaded_input) == ["time", "code", "alpha", "beta"]
    assert not any("use query" in script or "coreQuerySourceData =" in script for script in caller.scripts)


def test_empty_exposure_query_keeps_all_input_keys_for_dos_null_outputs(monkeypatch, sessions):
    source = raw_frame()
    caller = FakeSession(source, source.assign(alpha=np.nan, beta=np.nan))
    monkeypatch.setattr(
        module, "query", lambda request, **kwargs: SimpleNamespace(
            data=pd.DataFrame(columns=["time", "code", "circ_mv", "industry_l0"]),
        ),
    )
    result = module.preprocess(source, session=caller, columns=["alpha", "beta"], settings=SETTINGS)
    assert result.shape == (3, 2) and result.isna().all().all()
    exposure = next(value for name, value in caller.uploads[0].items() if name.endswith("Exposure"))
    assert len(exposure) == len(source) and exposure.industry_l0.isna().all()
    assert sessions[0].closed and not caller.closed


@pytest.mark.parametrize("use_variable", [False, True])
def test_partial_exposures_are_left_joined_without_falling_back_to_input(
    monkeypatch, sessions, use_variable,
):
    source = raw_frame().assign(circ_mv=999.0, industry_l0="do not fill missing exposures")
    response = raw_frame().assign(alpha=[0.25, np.nan, np.nan], beta=[-0.5, np.nan, np.nan])
    caller = FakeSession(source, response)
    monkeypatch.setattr(
        module, "query", lambda request, **kwargs: SimpleNamespace(data=raw_exposures().iloc[[0]]),
    )
    result = module.preprocess(
        DosVar("coreQueryData") if use_variable else source,
        session=caller, columns=["alpha", "beta"], settings=SETTINGS,
    )
    exposure = next(value for name, value in caller.uploads[0].items() if name.endswith("Exposure"))
    assert len(exposure) == len(source)
    assert exposure.loc[0, "circ_mv"] == 1.0
    assert exposure.loc[1:, ["circ_mv", "industry_l0"]].isna().all().all()
    pd.testing.assert_frame_equal(result, expected_panel(response, ["alpha", "beta"]))
    assert sessions[0].closed and not caller.closed


def test_independent_exposures_preserve_engine_symbol_input_keys(monkeypatch, sessions):
    source = raw_frame()
    source["code"] = source["code"].map(module.engine_symbol)
    caller = FakeSession(source)
    requests = []

    def query(request, *, session):
        requests.append(request)
        return SimpleNamespace(data=raw_exposures())

    monkeypatch.setattr(module, "query", query)
    module.preprocess(DosVar("coreQueryData"), session=caller, columns=["alpha", "beta"], settings=SETTINGS)
    exposure = next(value for name, value in caller.uploads[0].items() if name.endswith("Exposure"))
    assert exposure.code.tolist() == source.code.tolist()
    assert requests[0]["factors"] == ["circ_mv", "industry_l0"]
    assert sessions[0].closed and not caller.closed


@pytest.mark.parametrize("bad", ["x; undef all", "x.y", "x[0]", "select * from x", "x()", "", "x\n"])
def test_dosvar_rejects_non_identifiers_before_running_any_script(bad, sessions):
    caller = FakeSession(raw_frame())
    with pytest.raises(ValueError, match="single DolphinDB identifier"):
        module.preprocess(DosVar(bad), session=caller, columns=["alpha", "beta"])
    assert not caller.scripts and not caller.uploads and not sessions


@pytest.mark.parametrize("value", [True, 1 + 2j])
def test_dosvar_rejects_bool_and_complex_from_metadata_without_fetching_factors(value, sessions):
    source = raw_frame().assign(alpha=value)
    caller = FakeSession(source)
    with pytest.raises(ValueError, match="real numeric"):
        module.preprocess(DosVar("coreQueryData"), session=caller, columns=["alpha"])
    assert "schema(coreQueryData).colDefs" in caller.scripts
    assert "select time, code from coreQueryData" not in caller.scripts
    assert not caller.closed and not sessions


def test_dosvar_duplicate_normalized_keys_are_rejected(sessions):
    source = raw_frame().iloc[[0, 0]].copy()
    source["code"] = ["000001.SZ", "000001.XSHE"]
    caller = FakeSession(source)
    with pytest.raises(ValueError, match="duplicate time/code"):
        module.preprocess(DosVar("coreQueryData"), session=caller, columns=["alpha", "beta"])
    assert not caller.closed and not sessions


def test_dosvar_requires_explicit_session(sessions):
    with pytest.raises(ValueError, match="explicit session"):
        module.preprocess(DosVar("coreQueryData"), columns=["alpha", "beta"])
    assert not sessions


@pytest.mark.parametrize("columns,match", [
    ([], "empty"), (["alpha", "alpha"], "unique"), (["missing"], "missing factor"),
    (["time"], "collide"), (["circ_mv"], "collide"), (["industry_l0"], "collide"),
    (["stock_pool_member"], "collide"), ([""], "nonempty"), ("alpha", "sequence"),
    (None, "sequence"),
])
def test_selected_column_validation(columns, match, sessions):
    with pytest.raises(ValueError, match=match):
        module.preprocess(raw_frame(), columns=columns)
    assert not sessions


def test_internal_columns_are_required_and_never_inferred(sessions):
    with pytest.raises(TypeError, match="columns"):
        module.preprocess(raw_frame())
    assert not sessions


@pytest.mark.parametrize("change,match", [
    (lambda data: data.iloc[:0], "empty"),
    (lambda data: data.assign(alpha=np.nan, beta=np.nan), "all NULL"),
    (lambda data: pd.concat([data, data.iloc[[0]]]), "duplicate time/code"),
    (lambda data: data.assign(time=pd.NaT), "non-NULL"),
    (lambda data: data.assign(time=pd.Timestamp("2025-01-03 12:00")), "calendar dates"),
    (lambda data: data.assign(code=None), "nonempty strings"),
    (lambda data: data.assign(alpha=True), "real numeric"),
    (lambda data: data.rename(columns={"beta": "alpha"}), "unique"),
])
def test_dataframe_rejects_invalid_inputs_without_connecting(change, match, sessions):
    with pytest.raises(ValueError, match=match):
        module.preprocess(change(raw_frame()), columns=["alpha", "beta"])
    assert not sessions


def test_alias_duplicates_are_rejected(sessions):
    source = raw_frame().iloc[[0, 0]].copy()
    source["code"] = ["000001.SZ", "000001.XSHE"]
    with pytest.raises(ValueError, match="duplicate time/code"):
        module.preprocess(source, columns=["alpha", "beta"])
    assert not sessions


@pytest.mark.parametrize("name", ["rawFactor", "rawMarketValue", "rawIndustry", "currentDate", "result", "alpha_processed", "factor_preprocess_row_id", "momentum score"])
def test_named_factors_do_not_collide_with_dos_local_variables(name, sessions):
    source = raw_frame().rename(columns={"alpha": name})
    caller = FakeSession(source)
    result = module.preprocess(source, session=caller, columns=[name])
    assert list(result) == [name]
    assert not caller.closed


@pytest.mark.parametrize("use_owned", [False, True])
@pytest.mark.parametrize("failure", ["fail_process", "fail_upload"])
def test_failure_finally_cleans_only_owned_variables_and_closes_only_owned_session(
    monkeypatch, sessions, use_owned, failure,
):
    caller = FakeSession(raw_frame())
    setattr(caller, failure, True)

    create_query_session = module.create_session
    calls = 0

    @contextmanager
    def create(settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                yield caller
            finally:
                caller.close()
        else:
            with create_query_session(settings) as query_session:
                yield query_session

    if use_owned:
        monkeypatch.setattr(module, "create_session", create)
    with pytest.raises(RuntimeError, match="DOS failed|upload failed"):
        module.preprocess(
            raw_frame(), session=None if use_owned else caller,
            columns=["alpha", "beta"], settings=SETTINGS,
        )
    assert caller.closed is use_owned
    assert not any(name.startswith("soloFp") for name in caller.variables)
    assert "coreQueryData" in caller.variables and "coreQuerySourceData" in caller.variables
    assert "undef" in caller.scripts[-1]


def test_fake_session_rejects_unsupported_isdef():
    with pytest.raises(RuntimeError, match="does not support isDef"):
        FakeSession(raw_frame()).run('if (isDef("missing")) undef("missing", VAR)')


@pytest.mark.parametrize("use_variable", [False, True])
def test_successful_preprocess_returns_after_supported_cleanup(sessions, use_variable):
    caller = FakeSession(raw_frame())
    result = module.preprocess(
        DosVar("coreQueryData") if use_variable else raw_frame(), session=caller,
        columns=["alpha", "beta"],
    )
    assert len(result) == len(raw_frame())
    cleanup = caller.scripts[-1]
    assert "isDef" not in cleanup
    assert ' in objs().name) { undef(' in cleanup
    assert not any(name.startswith("soloFp") for name in caller.variables)
    assert set(caller.variables) == {"coreQueryData", "coreQuerySourceData"}
    assert not caller.closed
    assert caller.run(cleanup) is None


def test_cleanup_failure_never_masks_original_exception(sessions):
    caller = FakeSession(raw_frame())
    caller.fail_process = caller.fail_cleanup = True
    with pytest.raises(RuntimeError, match="DOS failed"):
        module.preprocess(raw_frame(), session=caller, columns=["alpha", "beta"])
    assert not caller.closed


def test_failed_exposure_query_closes_query_session_not_caller(monkeypatch, sessions):
    source = raw_frame()
    caller = FakeSession(source)

    def fail(*args, **kwargs):
        raise RuntimeError("exposure failed")

    monkeypatch.setattr(module, "query", fail)
    with pytest.raises(RuntimeError, match="exposure failed"):
        module.preprocess(DosVar("coreQueryData"), session=caller, columns=["alpha", "beta"], settings=SETTINGS)
    assert len(sessions) == 1 and sessions[0].closed
    assert not caller.closed and "undef" in caller.scripts[-1]


def test_result_must_preserve_keys_and_still_accept_all_null_result(sessions):
    source = raw_frame()
    response = source.assign(alpha=np.nan, beta=np.nan)
    caller = FakeSession(source, response)
    result = module.preprocess(source, session=caller, columns=["alpha", "beta"])
    assert result.shape == (3, 2) and result.isna().all().all()
    caller.response = response.iloc[:-1]
    with pytest.raises(ValueError, match="changed the input time/code keys"):
        module.preprocess(source, session=caller, columns=["alpha", "beta"])
    assert not any(name.startswith("soloFp") for name in caller.variables)


def test_each_call_uses_distinct_private_uuid_namespace(sessions):
    caller = FakeSession(raw_frame())
    module.preprocess(raw_frame(), session=caller, columns=["alpha", "beta"])
    module.preprocess(raw_frame(), session=caller, columns=["alpha", "beta"])
    assert set(caller.uploads[0]).isdisjoint(caller.uploads[1])


def test_factor_method_delegates_declared_columns_without_public_columns_argument(monkeypatch):
    calls = []
    algo_module = import_module("scheme.base.projects.factor.algo")
    def preprocess(data, **kwargs):
        calls.append((data, kwargs))
        return expected_panel(data, kwargs["columns"])

    monkeypatch.setattr(algo_module, "preprocess", preprocess)

    class MyFactor(Factor[FactorParams]):
        columns = ["alpha"]

        def compute(self, start, end):
            return self.preprocess(raw_frame()[["time", "code", "alpha"]], settings=SETTINGS)

    factor = MyFactor(FactorParams())
    factor.compute(None, None)
    assert calls[0][1] == {
        "session": None, "columns": ["alpha"], "market_value_column": "circ_mv",
        "industry_column": "industry_l0", "settings": SETTINGS,
    }


def test_inline_dos_matches_runtime_order_rank_and_null_contracts(sessions):
    # Inspect the submitted block; fixed responses never simulate DOS arithmetic.
    caller = FakeSession(raw_frame())
    module.preprocess(raw_frame(), session=caller, columns=["alpha", "beta"])
    source = re.sub(r"\bsoloFp[0-9a-f]{32}", "", caller.process_scripts[0])
    ordered = [
        'Working = sql(sqlCol(symbol(["time", "code"] join Factors)),',
        "replaceColumn!(Working, `time, timestamp(Working[`time]))",
        "replaceColumn!(Working, `code, string(Working[`code]))",
        "Working = lj(Working, Exposure, `time`code)",
        "Working.sortBy!(`time`code)",
        "Output = select time, code from Working",
        "Output[Selected] = take(double(NULL), Working.rows())",
        'if (!HasValues) throw "preprocess factors are all NULL"',
        "DayRows = at(Working[`time] == Day)",
        "CrossSection = Working[Working[`time] == Day]",
        "for (Selected in Factors)", "Raw = double(", "RawMarketValue = double(",
        "RawIndustry = string(",
        "ValidMask = isValid(Raw) && isValid(RawMarketValue) && isValid(RawIndustry)",
        "MarketValue = iif(MarketValue < 1, 1.0, MarketValue)",
        "MvValues = log(MarketValue)", "MvStd = stdp(MvValues)",
        "MvValues = (MvValues - avg(MvValues)) \\ MvStd",
        "Y = Raw[ValidMask]", "Median = median(Y)", "Mad = median(abs(Y - Median))",
        "if (isValid(Mad) && Mad > 0) {",
        "Lower = Median - 3.0 * Mad", "Upper = Median + 3.0 * Mad",
        "Y = iif(Y < Lower, Lower, iif(Y > Upper, Upper, Y))", "YStd = stdp(Y)",
        "Y = (Y - avg(Y)) \\ YStd",
        "RegressionMask = isValid(Y) && isValid(MvValues)",
        "RowIds = DayRows[ValidMask][RegressionMask]",
        "Industries = sort(distinct(IndustryValues))",
        "for (Index in 1 .. (size(Industries) - 1))",
        "Design = matrix(take(1.0, size(Y))) join X",
        "SingularValues = svd(Design, false, false)",
        "RankTolerance = max([rows(Design), cols(Design)]) * max(SingularValues) * 2.220446049250313e-16",
        "EffectiveRank = sum(SingularValues > RankTolerance)",
        "if (size(Y) <= EffectiveRank) continue", 'Beta = ols(Y, X, true, 0, "svd")',
        "Fitted = Beta[0] + Beta[1] * MvValues",
        "if (size(Industries) > 1) {", "for (Index in 1 .. (size(Industries) - 1))",
        "Fitted = Fitted + Beta[Index + 1] * flatten(X[, Index])", "RawResidual = Y - Fitted",
        "ResidualTolerance = max(abs(Y)) * max([rows(Design), cols(Design)]) * 2.220446049250313e-16",
        "ResidualStd = stdp(RawResidual)", "ResidualStd <= ResidualTolerance",
        "Residual = (RawResidual - avg(RawResidual)) \\ ResidualStd",
        "ProcessedValues[RowIds] = Residual", "Output[Selected] = ProcessedValues",
    ]
    position = 0
    for snippet in ordered:
        position = source.index(snippet, position) + len(snippet)
    assert source.count("stdp(") == 3
    assert source.count("2.220446049250313e-16") == 2
    assert all(forbidden not in source for forbidden in (
        "def ", "factor::", "common::", "ta::", "use ", "nGroups", '"_group"', "1.4826",
        "Working[`time] = timestamp(", "Working[`code] = string(",
        "med(", ".columns()", "max([1.0,",
        "Beta[size(Beta) - 1]",
    ))


def test_inline_current_date_factor_and_all_temporaries_have_private_names(sessions):
    frame = raw_frame().rename(columns={"alpha": "currentDate"})
    caller = FakeSession(frame)
    module.preprocess(frame, session=caller, columns=["currentDate"])
    source = caller.process_scripts[0]
    prefix = re.search(r"soloFp[0-9a-f]{32}", source).group()
    assert f"CrossSection = {prefix}Working[{prefix}Working[`time] == {prefix}Day]" in source
    assert "currentDate" not in source
    assert "where" not in source and "_$" not in source
    assignments = re.findall(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*=", source, flags=re.MULTILINE)
    loops = re.findall(r"for \(([A-Za-z][A-Za-z0-9]*) in", source)
    assert assignments and loops
    assert all(name.startswith(prefix) for name in assignments + loops)
    created = set(re.findall(r"\bsoloFp[0-9a-f]{32}[A-Za-z]+\b", source))
    cleaned = set(re.findall(r'undef\("([A-Za-z0-9_]+)", VAR\)', caller.scripts[-1]))
    assert created | set(caller.uploads[0]) <= cleaned
    assert all(name.startswith(prefix) for name in cleaned)


def test_preprocess_has_no_resource_loader_or_single_use_wrappers():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert all(forbidden not in source for forbidden in (
        "scheme.data.dolphindb.scripts", "factor_script", "factor_helpers.dos",
        "factor_preprocess.dos", "soloFactorPreprocess", "def _exposures", "def _canonical",
    ))
