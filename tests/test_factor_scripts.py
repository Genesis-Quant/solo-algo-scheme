"""Offline checks for session-local factor functions vendored from the pinned Runtime."""

import ast
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from collections import Counter
from contextlib import nullcontext
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from scheme.config import DolphinSettings
from scheme.data.dolphindb import scripts
from scheme.execute.factor import api

HELPERS = (
    "factorStringVector", "factorCheckColumns", "factorWeightedReturn", "factorExtremeRanks",
    "factorExtremeWeightedReturn", "factorValidateGroups", "factorZScore", "factorClipMad",
)
ANALYSIS = (
    "factorFilterNulls", "factorExecutionStatistics", "factorInformationCoefficient",
    "factorGroupReturns", "factorGroupTurnover",
)
RENAMES = {name: "solo" + name[0].upper() + name[1:] for name in HELPERS + ANALYSIS}
OLD_FUNCTIONS = re.compile(r"\b(?:" + "|".join(RENAMES) + r")\b")
RESOURCES = files("scheme.data.dolphindb")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime_definitions():
    # Read literals with AST, never import runtime.database.compile or its dotenv side effects.
    source = distribution("arena-runtime").locate_file("runtime/database/compile/factor/functions")
    definitions = {}
    constants = {}
    for filename in ("helpers.py", "preprocess.py", "statistics.py"):
        tree = ast.parse((source / filename).read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            fields = {item.arg: item.value for item in node.value.keywords}
            if "definition" not in fields:
                continue
            definition = textwrap.dedent(ast.literal_eval(fields["definition"])).strip()
            name = re.search(r"\bdef\s+(\w+)\s*\(", definition)[1]
            dependencies = fields.get("dependencies", ast.Tuple(elts=[]))
            definitions[name] = definition, [item.id for item in dependencies.elts]
            constants[node.targets[0].id] = name
    return definitions, constants


@pytest.mark.parametrize("kind,names", [("helpers", HELPERS), ("analysis", ANALYSIS)])
def test_resources_preserve_pinned_definitions_except_whitelisted_function_names(
    runtime_definitions, kind, names,
):
    definitions, _ = runtime_definitions
    expected = "\n\n".join(
        OLD_FUNCTIONS.sub(lambda match: RENAMES[match[0]], definitions[name][0])
        for name in names
    ) + "\n"
    actual = RESOURCES.joinpath(f"factor_{kind}.dos").read_text(encoding="utf-8")
    assert actual == expected
    # Keep all quoted field names, output names and error text unchanged as well.
    literals = r'''"[^"]*"|'[^']*' '''.strip()
    assert re.findall(literals, actual) == re.findall(
        literals, "\n".join(definitions[name][0] for name in names),
    )


def test_analysis_dependency_closure_contains_all_helpers_and_no_old_symbols(runtime_definitions):
    definitions, constants = runtime_definitions
    required = set()

    def visit(name):
        if name in required:
            return
        required.add(name)
        for dependency in definitions[name][1]:
            visit(constants[dependency])

    for name in ANALYSIS:
        visit(name)
    assert required == set(ANALYSIS + HELPERS[:-2])
    script = scripts.factor_script("analysis")
    defined = Counter(re.findall(r"\bdef\s+(soloFactor\w+)\s*\(", script))
    assert defined == Counter(RENAMES.values())
    assert set(re.findall(r"\bsoloFactor[A-Z]\w*\b", script)) == set(defined)
    assert not OLD_FUNCTIONS.search(script)
    assert "::" not in script
    assert not re.search(r"(?m)^\s*(?:use|module)\b", script)
    helpers = RESOURCES.joinpath("factor_helpers.dos").read_text(encoding="utf-8")
    assert set(re.findall(r"\bdef\s+(\w+)\s*\(", helpers)) == {RENAMES[n] for n in HELPERS}
    assert not OLD_FUNCTIONS.search(helpers)
    assert script.startswith(helpers + "\n")
    for name in ANALYSIS:
        for helper in HELPERS:
            assert script.index("def " + RENAMES[helper]) < script.index("def " + RENAMES[name])


def test_make_call_uses_renamed_function_objects_not_runtime_namespace():
    script = scripts.factor_script("analysis")
    assert re.findall(r"\bmakeCall\s*\(\s*(\w+)", script) == [
        "corr", "spearmanr", "soloFactorExtremeWeightedReturn",
        "soloFactorExtremeWeightedReturn", "soloFactorWeightedReturn",
    ]


@pytest.mark.parametrize("missing", ["factor_helpers.dos", "factor_analysis.dos"])
def test_missing_resource_fails_without_module_fallback(monkeypatch, tmp_path, missing):
    for name in ("factor_helpers.dos", "factor_analysis.dos"):
        if name != missing:
            tmp_path.joinpath(name).write_text("def untouched() {}", encoding="utf-8")
    monkeypatch.setattr(scripts, "files", lambda _: tmp_path)
    with pytest.raises(FileNotFoundError) as error:
        scripts.factor_script("analysis")
    assert Path(error.value.filename).name == missing


@pytest.fixture
def analysis_run(monkeypatch):
    parameters = api.FactorAnalysisParams(
        start="2025-01-02", end="2025-01-06", columns=["score"], period=5,
        price_column="close_hfq", max_lag=3,
    )
    panel = pd.DataFrame(
        {"score": [1.0, 2.0]},
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2025-01-02"]), ["000001.XSHE", "600000.XSHG"]],
            names=["date", "symbol"],
        ),
    )
    factor = SimpleNamespace(compute=Mock(return_value=panel))
    session = SimpleNamespace(run=Mock(return_value=pd.DataFrame()), upload=Mock())
    monkeypatch.setattr(api, "create_session", lambda _: nullcontext(session))
    monkeypatch.setattr(
        api, "query", Mock(return_value=nullcontext(SimpleNamespace(data_ref="offlineReturns"))),
    )

    def run():
        return api.analyze_factors(
            factor, parameters,
            settings=DolphinSettings(host="unused.invalid", username="offline", password="unused"),
        )

    return run, session


def test_analysis_inlines_resource_verbatim_before_setup(monkeypatch, analysis_run):
    run, session = analysis_run
    loader = Mock(wraps=scripts.factor_script)
    monkeypatch.setattr(api, "factor_script", loader)
    report = run()
    loader.assert_called_once_with("analysis")
    calls = [call.args[0] for call in session.run.call_args_list]
    assert calls[0].startswith(scripts.factor_script("analysis"))
    assert "soloCalendar = exec time from offlineReturns where code = soloCalendarSymbol" in calls[0]
    assert "soloFactorProcessed = soloFactorFilterNulls(soloFactorInput, soloFactors)" in calls[0]
    assert calls[1:] == [
        "soloFactorProcessed",
        "soloFactorExecutionStatistics(soloFactorSource, soloFactorInput, "
        'soloFactorProcessed, array(STRING, 0), soloStart, soloEnd, "time", "code")',
        'soloFactorInformationCoefficient(soloFactorProcessed, soloReturns, soloFactors, "time")',
        "soloFactorGroupReturns(soloFactorProcessed, soloReturns, soloFactors, soloGroups, "
        'soloSelect, "time", "code", "circ_mv")',
        "soloFactorGroupTurnover(soloFactorProcessed, soloFactors, soloPeriods, soloGroups, "
        'soloSelect, "time", "code", soloCalendar)',
    ]
    assert not re.search(r"(?m)^\s*(?:use|module)\b|::", "\n".join(calls))
    assert report.parameters.return_columns == ["P5L1", "P5L2", "P5L3"]


def test_analysis_does_not_mask_missing_resources(monkeypatch, analysis_run):
    run, session = analysis_run
    failure = FileNotFoundError("factor_analysis.dos is missing")
    monkeypatch.setattr(api, "factor_script", Mock(side_effect=failure))
    with pytest.raises(FileNotFoundError) as error:
        run()
    assert error.value is failure
    session.run.assert_not_called()


def test_analysis_definition_failure_is_not_retried_or_replaced(analysis_run):
    run, session = analysis_run
    failure = RuntimeError("session function definition failed")
    session.run.side_effect = failure
    with pytest.raises(RuntimeError) as error:
        run()
    assert error.value is failure
    session.run.assert_called_once()
    assert session.run.call_args.args[0].startswith(scripts.factor_script("analysis"))


def test_inline_dos_resources_are_packaged_verbatim_in_wheel_and_sdist(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the offline packaging check")
    result = subprocess.run(
        [uv, "build", str(ROOT), "--out-dir", str(tmp_path), "--offline",
         "--no-create-gitignore", "--no-python-downloads", "--python", sys.executable],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        path.name: path.read_bytes()
        for path in ROOT.joinpath("src/scheme/data/dolphindb").glob("*.dos")
    }
    assert {"factor_helpers.dos", "factor_analysis.dos", "factor_preprocess.dos", "arena_backtest.dos"} <= expected.keys()
    with zipfile.ZipFile(next(tmp_path.glob("*.whl"))) as wheel:
        for name, content in expected.items():
            assert wheel.read(f"scheme/data/dolphindb/{name}") == content
    with tarfile.open(next(tmp_path.glob("*.tar.gz"))) as sdist:
        prefix = sdist.getmembers()[0].name.split("/")[0]
        for name, content in expected.items():
            with sdist.extractfile(f"{prefix}/src/scheme/data/dolphindb/{name}") as resource:
                assert resource.read() == content
