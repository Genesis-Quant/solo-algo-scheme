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
        "circ_mv": [1.0, 4.0, 6.0], "industry_l0": ["bank", "bank", None],
    })


class FakeSession:
    """Only transports fixed server responses; deliberately does not neutralize."""

    def __init__(self, source, response=None):
        self.source = source.copy(deep=True)
        self.response = response.copy(deep=True) if response is not None else source.copy(deep=True)
        self.scripts = []
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
        if script.startswith("if (isDef("):
            if self.fail_cleanup:
                raise RuntimeError("cleanup failed")
            for name in re.findall(r'undef\("([A-Za-z0-9_]+)", VAR\)', script):
                self.variables.pop(name, None)
            return None
        if " = soloFactorPreprocess(" in script:
            if self.fail_process:
                raise RuntimeError("DOS failed")
            self.variables[script.split(" = ")[0]] = self.response.copy(deep=True)
            return None
        if re.fullmatch(r"soloFp[0-9a-f]{32}Result", script):
            return self.variables[script]
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

    monkeypatch.setattr(module, "create_session", create)
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
    assert list(input_frame) == ["time", "code", "beta", "alpha", "circ_mv", "industry_l0"]
    assert all(re.fullmatch(r"soloFp[0-9a-f]{32}[A-Za-z]+", name) for name in uploaded)
    assert any("def soloFactorPreprocess(" in script for script in session.scripts)
    assert any(" = soloFactorPreprocess(" in script for script in session.scripts)
    assert session.variables == {"coreQueryData": session.source, "coreQuerySourceData": "caller cache"}
    assert not session.closed and not sessions


def test_dataframe_owned_session_closed_and_default_columns_exclude_exposures(sessions):
    result = module.preprocess(raw_frame(), settings=SETTINGS)
    assert list(result) == ["alpha", "beta"]
    assert len(sessions) == 1 and sessions[0].closed
    assert not any(name.startswith("soloFp") for name in sessions[0].variables)


def test_from_env_is_only_used_for_owned_session_and_does_not_load_dotenv(sessions):
    module.preprocess(raw_frame())
    assert len(sessions) == 1 and sessions[0].closed


def test_canonical_input_is_supported_without_modification(sessions):
    canonical = expected_panel(raw_frame(), ["alpha", "beta"])
    canonical["circ_mv"] = [4.0, 6.0, 1.0]
    canonical["industry_l0"] = ["bank", "bank", "bank"]
    original = canonical.copy(deep=True)
    session = FakeSession(raw_frame())
    result = module.preprocess(canonical, session=session)
    assert list(result.index.names) == ["date", "symbol"]
    pd.testing.assert_frame_equal(canonical, original)
    uploaded = next(value for name, value in session.uploads[0].items() if name.endswith("Input"))
    assert set(uploaded.code) == {"000001.SZ", "600000.SH"}


def test_dosvar_borrows_session_downloads_only_keys_and_never_mutates_original(sessions):
    source = raw_frame()
    source["stock_pool_member"] = True
    original = source.copy(deep=True)
    session = FakeSession(source)
    result = module.preprocess(DosVar("coreQueryData"), session=session)
    pd.testing.assert_frame_equal(result, expected_panel(source, ["alpha", "beta"]))
    pd.testing.assert_frame_equal(session.source, original)
    assert not session.closed and not sessions
    assert "select time, code from coreQueryData" in session.scripts
    assert any(" = coreQueryData.copy(); " in script for script in session.scripts)
    assert all(not key.endswith("Input") for key in session.uploads[0])
    cleanup = session.scripts[-1]
    assert "coreQueryData" not in cleanup and "coreQuerySourceData" not in cleanup
    assert not any("use factor" in script or "factor::" in script for script in session.scripts)


@pytest.mark.parametrize("use_variable", [False, True])
def test_missing_exposures_query_exact_input_range_codes_in_separate_owned_session(
    monkeypatch, sessions, use_variable,
):
    source = raw_frame().drop(columns="industry_l0")
    source["circ_mv"] = [np.nan, 4.0, 6.0]  # Existing NULLs must not be backfilled.
    caller = FakeSession(source)
    calls = []

    def query(request, *, session):
        validated = FactorQuery.model_validate(request)
        assert validated.lookback.total_seconds() == 0
        assert session is not caller
        assert not session.closed
        assert "coreQuerySourceData" not in request
        calls.append((request, session))
        exposure = raw_frame()[["time", "code", "industry_l0"]]
        # Extra dates/codes from a provider cannot expand the output keys.
        exposure = pd.concat([exposure, pd.DataFrame({
            "time": pd.to_datetime(["2025-01-02", "2025-01-07"]),
            "code": ["999999.SZ", "000001.SZ"], "industry_l0": ["other", "future"],
        })], ignore_index=True)
        return SimpleNamespace(data=exposure)

    monkeypatch.setattr(module, "query", query)
    result = module.preprocess(
        DosVar("coreQueryData") if use_variable else source,
        session=caller, settings=SETTINGS,
    )
    assert len(result) == len(source)
    request, owned = calls[0]
    assert request == {
        "start_date": "2025-01-03", "end_date": "2025-01-06",
        "codes": ["000001.SZ", "600000.SH"], "factors": ["industry_l0"],
    }
    assert owned.closed and not caller.closed
    uploaded = next(value for name, value in caller.uploads[0].items() if name.endswith("Exposure"))
    assert list(uploaded) == ["time", "code", "industry_l0"]
    assert uploaded.time.tolist() == source.time.tolist()
    assert any(" = lj(" in script for script in caller.scripts)
    assert not any("use query" in script or "coreQuerySourceData =" in script for script in caller.scripts)


def test_empty_exposure_query_keeps_all_input_keys_for_dos_null_outputs(monkeypatch, sessions):
    source = raw_frame().drop(columns="industry_l0")
    caller = FakeSession(source, source.assign(alpha=np.nan, beta=np.nan))
    monkeypatch.setattr(
        module, "query", lambda request, **kwargs: SimpleNamespace(
            data=pd.DataFrame(columns=["time", "code", "industry_l0"]),
        ),
    )
    result = module.preprocess(source, session=caller, settings=SETTINGS)
    assert result.shape == (3, 2) and result.isna().all().all()
    exposure = next(value for name, value in caller.uploads[0].items() if name.endswith("Exposure"))
    assert len(exposure) == len(source) and exposure.industry_l0.isna().all()
    assert sessions[0].closed and not caller.closed


def test_missing_both_exposures_preserves_engine_symbol_input_keys(monkeypatch, sessions):
    source = raw_frame().drop(columns=["circ_mv", "industry_l0"])
    source["code"] = source["code"].map(module.engine_symbol)
    caller = FakeSession(source)
    requests = []

    def query(request, *, session):
        requests.append(request)
        return SimpleNamespace(data=raw_frame()[["time", "code", "circ_mv", "industry_l0"]])

    monkeypatch.setattr(module, "query", query)
    module.preprocess(DosVar("coreQueryData"), session=caller, settings=SETTINGS)
    exposure = next(value for name, value in caller.uploads[0].items() if name.endswith("Exposure"))
    assert exposure.code.tolist() == source.code.tolist()
    assert requests[0]["factors"] == ["circ_mv", "industry_l0"]
    assert sessions[0].closed and not caller.closed


@pytest.mark.parametrize("bad", ["x; undef all", "x.y", "x[0]", "select * from x", "x()", "", "x\n"])
def test_dosvar_rejects_non_identifiers_before_running_any_script(bad, sessions):
    caller = FakeSession(raw_frame())
    with pytest.raises(ValueError, match="single DolphinDB identifier"):
        module.preprocess(DosVar(bad), session=caller)
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
        module.preprocess(DosVar("coreQueryData"), session=caller)
    assert not caller.closed and not sessions


def test_dosvar_requires_explicit_session(sessions):
    with pytest.raises(ValueError, match="explicit session"):
        module.preprocess(DosVar("coreQueryData"))
    assert not sessions


@pytest.mark.parametrize("columns,match", [
    ([], "empty"), (["alpha", "alpha"], "unique"), (["missing"], "missing factor"),
    (["time"], "collide"), (["circ_mv"], "collide"), (["industry_l0"], "collide"),
    (["stock_pool_member"], "collide"), ([""], "nonempty"), ("alpha", "sequence"),
])
def test_selected_column_validation(columns, match, sessions):
    with pytest.raises(ValueError, match=match):
        module.preprocess(raw_frame(), columns=columns)
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
        module.preprocess(change(raw_frame()))
    assert not sessions


def test_alias_duplicates_are_rejected(sessions):
    source = raw_frame().iloc[[0, 0]].copy()
    source["code"] = ["000001.SZ", "000001.XSHE"]
    with pytest.raises(ValueError, match="duplicate time/code"):
        module.preprocess(source)
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

    @contextmanager
    def create(settings):
        try:
            yield caller
        finally:
            caller.close()

    if use_owned:
        monkeypatch.setattr(module, "create_session", create)
    with pytest.raises(RuntimeError, match="DOS failed|upload failed"):
        module.preprocess(raw_frame(), session=None if use_owned else caller, settings=SETTINGS)
    assert caller.closed is use_owned
    assert not any(name.startswith("soloFp") for name in caller.variables)
    assert "coreQueryData" in caller.variables and "coreQuerySourceData" in caller.variables
    assert "undef" in caller.scripts[-1]


def test_cleanup_failure_never_masks_original_exception(sessions):
    caller = FakeSession(raw_frame())
    caller.fail_process = caller.fail_cleanup = True
    with pytest.raises(RuntimeError, match="DOS failed"):
        module.preprocess(raw_frame(), session=caller)
    assert not caller.closed


def test_failed_exposure_query_closes_query_session_not_caller(monkeypatch, sessions):
    source = raw_frame().drop(columns="industry_l0")
    caller = FakeSession(source)

    def fail(*args, **kwargs):
        raise RuntimeError("exposure failed")

    monkeypatch.setattr(module, "query", fail)
    with pytest.raises(RuntimeError, match="exposure failed"):
        module.preprocess(DosVar("coreQueryData"), session=caller, settings=SETTINGS)
    assert len(sessions) == 1 and sessions[0].closed
    assert not caller.closed and "undef" in caller.scripts[-1]


def test_result_must_preserve_keys_and_still_accept_all_null_result(sessions):
    source = raw_frame()
    response = source.assign(alpha=np.nan, beta=np.nan)
    caller = FakeSession(source, response)
    result = module.preprocess(source, session=caller)
    assert result.shape == (3, 2) and result.isna().all().all()
    caller.response = response.iloc[:-1]
    with pytest.raises(ValueError, match="changed the input time/code keys"):
        module.preprocess(source, session=caller)
    assert not any(name.startswith("soloFp") for name in caller.variables)


def test_each_call_uses_distinct_private_uuid_namespace(sessions):
    caller = FakeSession(raw_frame())
    module.preprocess(raw_frame(), session=caller)
    module.preprocess(raw_frame(), session=caller)
    assert set(caller.uploads[0]).isdisjoint(caller.uploads[1])


def test_factor_method_delegates_the_complete_public_signature(monkeypatch):
    calls = []
    algo_module = import_module("scheme.base.projects.factor.algo")
    monkeypatch.setattr(algo_module, "preprocess", lambda data, **kwargs: calls.append((data, kwargs)))

    class MyFactor(Factor[FactorParams]):
        def compute(self, start, end):
            return self.preprocess(raw_frame(), columns=["alpha"], settings=SETTINGS)

    factor = MyFactor(FactorParams())
    factor.compute(None, None)
    assert calls[0][1] == {
        "session": None, "columns": ["alpha"], "market_value_column": "circ_mv",
        "industry_column": "industry_l0", "settings": SETTINGS,
    }


def test_dos_algorithm_source_matches_runtime_order_rank_and_null_contracts():
    # Structural contracts only. Actual numerical execution belongs to synthetic DDB tests.
    source = Path(module.__file__).parents[3].joinpath("data/dolphindb/factor_preprocess.dos").read_text(encoding="utf-8")
    ordered = [
        "workingTable.sortBy!(symbol([timeCol, codeCol]))",
        "crossSection = workingTable[workingTable[timeCol] == currentDate]",
        "crossSection.sortBy!(symbol([codeCol]))", "for (factorCol in factorColNames)",
        "rawFactor = double(", "rawMarketValue = double(", "rawIndustry = string(",
        "validMask = isValid(rawFactor) && isValid(rawMarketValue) && isValid(rawIndustry)",
        "marketValue = iif(marketValue < 1, 1.0, marketValue)",
        "mv = soloFactorZScore(log(marketValue))", "y = soloFactorZScore(soloFactorClipMad(rawFactor[validMask]))",
        "regressionMask = isValid(y) && isValid(mv)", "industries = sort(distinct(industryValues))",
        "for (i in 1 .. (size(industries) - 1))", "design = matrix(take(1.0, size(y))) join x",
        "singularValues = svd(design, false, false)", "effectiveRank = sum(singularValues > rankTolerance)",
        "if (size(y) <= effectiveRank) continue", 'beta = ols(y, x, true, 0, "svd")',
        "rawResidual = y - fitted", "stdp(rawResidual) <= residualTolerance",
        "residual = soloFactorZScore(rawResidual)", "processedValues[rowIds] = residual",
    ]
    position = 0
    for snippet in ordered:
        position = source.index(snippet, position) + len(snippet)
    assert "2.220446049250313e-16" in source
    assert 'result[factorCol] = take(double(NULL), workingTable.rows())' in source
    assert all(forbidden not in source for forbidden in ("factor::", "common::", "ta::", "use ", "nGroups", '"_group"'))
    assert "1.4826" not in source


def test_dos_current_date_named_factor_cannot_shadow_cross_section_date():
    # This is a resource contract, not a Python simulation of DolphinDB evaluation.
    source = module.factor_script("preprocess")
    assert "for (currentDate in dates)" in source
    assert "crossSection = workingTable[workingTable[timeCol] == currentDate]" in source
    assert "where _$timeCol = currentDate" not in source
    assert "_$currentDate" not in source
