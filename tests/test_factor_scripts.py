"""Offline checks for the DOS actually submitted by the factor API."""

import ast
import inspect
import re
from contextlib import nullcontext
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from scheme.config import DolphinSettings
from scheme.execute.factor import api

ROOT_FUNCTIONS = (
    "soloFactorFilterNulls", "soloFactorExecutionStatistics", "soloFactorInformationCoefficient",
    "soloFactorGroupReturns", "soloFactorGroupTurnover",
)


@pytest.fixture
def analysis_run(monkeypatch):
    parameters = api.FactorAnalysisParams(
        start="2025-01-02", end="2025-01-06", period=5,
        price_column="close_hfq", max_lag=3,
    )
    panel = pd.DataFrame(
        {"score": [1.0, 2.0]},
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2025-01-02"]), ["000001.XSHE", "600000.XSHG"]],
            names=["date", "symbol"],
        ),
    )
    factor = SimpleNamespace(columns=("score",), compute=Mock(return_value=panel))
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


def test_analysis_runs_inline_steps_then_downloads_assigned_outputs(analysis_run):
    run, session = analysis_run
    report = run()
    calls = [call.args[0] for call in session.run.call_args_list]
    script = calls[0]
    steps = [
        "soloCalendar = exec time from offlineReturns where code = soloCalendarSymbol",
        "soloFactorInput = lj(soloFactorSource, soloPrices, `time`code)",
        "soloFactorProcessed = soloFactorInput[soloValidMask]",
        "soloOutputExecutionStatistics = select count(*) as source_count",
        "soloOutputInformationCoefficient = select distinct(time) as time",
        "soloOutputGroupReturns = select distinct(time) as time",
        "soloOutputGroupTurnover = table(",
        "soloOutputGroupTurnover.sortBy!(`factor`periods`time)",
    ]
    assert all(step in script for step in steps)
    assert [script.index(step) for step in steps] == sorted(script.index(step) for step in steps)
    assert calls[1:] == [
        "soloFactorProcessed", "soloOutputExecutionStatistics", "soloOutputInformationCoefficient",
        "soloOutputGroupReturns", "soloOutputGroupTurnover",
    ]
    assert all(name not in script for name in ROOT_FUNCTIONS)
    assert not re.search(r"(?m)^\s*(?:use|module)\b|::", "\n".join(calls))
    assert report.parameters.columns == ["score"]
    assert report.parameters.return_columns == ["P5L1", "P5L2", "P5L3"]
    uploaded = session.upload.call_args.args[0]
    assert uploaded["soloFactors"].tolist() == ["score"]
    assert uploaded["soloFactorSource"].columns.tolist() == ["time", "code", "score"]


def test_only_repeated_aggregation_helpers_are_defined(analysis_run):
    run, session = analysis_run
    run()
    script = session.run.call_args_list[0].args[0]
    assert re.findall(r"\bdef\s+(\w+)\s*\(", script) == [
        "soloFactorWeightedReturn", "soloFactorExtremeRanks", "soloFactorExtremeWeightedReturn",
    ]
    assert re.findall(r"\bmakeCall\s*\(\s*(\w+)", script) == [
        "corr", "spearmanr", "soloFactorExtremeWeightedReturn",
        "soloFactorExtremeWeightedReturn", "soloFactorWeightedReturn",
    ]
    assert "soloFactorValidateGroups" not in script
    assert "soloFactorStringVector" not in script
    assert "soloFactorCheckColumns" not in script
    assert "soloFactorZScore" not in script
    assert "soloFactorClipMad" not in script


def test_inline_factor_filter_weights_and_extremes_preserve_null_and_tie_rules(analysis_run):
    run, session = analysis_run
    run()
    script = session.run.call_args_list[0].args[0]
    assert "soloValidMask = soloValidMask && !isNull(soloFactorInput[soloFactorCol])" in script
    assert "rank(_$soloFactorCol, true, , true, `first, false) * soloGroups" in script
    assert "double(count(_$soloFactorCol))" in script
    assert "valid = !isNull(weight) && !isNull(ret)" in script
    assert "if (!isValid(totalWeight) || totalWeight <= 0)" in script
    assert "return sum(validWeight * validReturn) \\ totalWeight" in script
    assert "double(values) as factor_value" in script
    assert "ranked.sortBy!(`factor_value`factor_code, [ascending, ascending])" in script
    assert "soloFactorExtremeRanks(factorValue, codes, ascending) < int(nSelect)" in script
    assert "soloFactorWeightedReturn(weight[selected], ret[selected])" in script
    assert script.index("soloFactorProcessed = soloFactorInput[soloValidMask]") < script.index(
        "if (any(isNull(soloFactorProcessed.circ_mv) || soloFactorProcessed.circ_mv <= 0))",
    )


def test_inline_statistics_keep_closed_boundaries_and_full_calendar(analysis_run):
    run, session = analysis_run
    run()
    script = session.run.call_args_list[0].args[0]
    assert script.count("where time >= soloStart, time < soloEnd, !isNull(code)") == 2
    assert 'soloOutputExecutionStatistics["filter0_name"]' in script
    assert 'soloOutputExecutionStatistics["filter0_count"]' in script
    assert '"因子空值", soloOutputExecutionStatistics.rows()' in script
    assert "sort(distinct(timestamp(soloCalendar)))" in script
    assert "sort(distinct(int(soloPeriods)))" in script
    assert "move(time, soloPeriod) as previous_time" in script
    assert "rank(factor_value, true, , true, `average, false) as factor_rank" in script
    assert "1.0 - sum(iif(isNull(retained), 0, 1)) \\ count(*) as turnover" in script
    assert "distinct(soloFactorSource.time)" not in script
    # Top-level variables are session-owned and must not shadow SQL columns such as factor.
    assignments = re.findall(r"(?m)^\s*([A-Za-z_]\w*)\s*=(?!=)", script)
    # These assignments are confined to the three reusable helper definitions.
    helper_locals = {
        "valid", "validWeight", "validReturn", "totalWeight", "ranks", "ranked", "selected",
    }
    assert all(name.startswith("solo") or name in helper_locals for name in assignments)


def test_analysis_does_not_load_resources_or_read_detached_dos(monkeypatch, analysis_run):
    source = inspect.getsource(api)
    assert "factor_script" not in source
    assert "scheme.data.dolphindb.scripts" not in source
    assert "importlib.resources" not in source
    assert "read_text" not in source
    assert ".dos" not in source
    tree = ast.parse(source)
    run_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "session"
        and node.func.attr == "run"
    ]
    assert len(run_calls) == 6
    assert all(isinstance(call.args[0], (ast.JoinedStr, ast.Constant)) for call in run_calls)
    fail = Mock(side_effect=AssertionError("factor analysis must not read DOS resources"))
    monkeypatch.setattr(resources, "files", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    run, session = analysis_run
    run()
    fail.assert_not_called()
    assert session.run.call_count == 6


def test_inline_analysis_failure_is_not_retried_or_replaced(analysis_run):
    run, session = analysis_run
    failure = RuntimeError("inline factor statistics failed")
    session.run.side_effect = failure
    with pytest.raises(RuntimeError) as error:
        run()
    assert error.value is failure
    session.run.assert_called_once()
    script = session.run.call_args.args[0]
    assert "soloFactorProcessed = soloFactorInput[soloValidMask]" in script
    assert "soloOutputGroupTurnover = table(" in script
