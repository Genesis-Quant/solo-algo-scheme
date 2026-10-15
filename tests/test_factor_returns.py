import json
import re
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from runtime.apps.query import FactorQuery

from scheme.base import FactorAnalysisParams, FactorReportForm
from scheme.execute.factor.api import _return_derivatives
from scheme.execute.schema import FactorTask

DATES = {"start": "2025-01-03", "end": "2025-01-09"}


def evaluate_returns(prices, max_lag):
    frame = prices.sort_values(["code", "time"]).copy()

    def evaluate(value):
        if isinstance(value, str):
            return frame[value]
        if not isinstance(value, dict):
            return value
        fields = value["fields"]
        if value["op"] == "unary.shift":
            return frame.groupby("code", sort=False)[fields["col"]].shift(value["params"]["periods"])
        left, right = evaluate(fields["left"]), evaluate(fields["right"])
        if value["op"] == "binary.mul":
            return left * right
        if value["op"] == "binary.div":
            return left / right
        assert value["op"] == "binary.sub"
        return left - right

    for name, node in _return_derivatives(max_lag).items():
        frame[name] = evaluate(node)
    return frame


@pytest.mark.parametrize("max_lag,expected", [
    (1, ["r0o", "r1"]),
    (2, ["r0o", "r1", "r1o", "r2"]),
    (3, ["r0o", "r1", "r1o", "r2", "r2o", "r3"]),
])
def test_lag_parameters_generate_ordered_segment_names_and_metadata(max_lag, expected):
    params = FactorAnalysisParams(**DATES, columns=["score"], max_lag=max_lag)
    assert params.return_columns == expected
    assert list(params.return_specs) == expected
    for lag in range(1, max_lag + 1):
        overnight, intraday = params.return_columns[2 * (lag - 1):2 * lag]
        assert params.return_specs[overnight] == {
            "kind": "simple", "periods": 1, "lag": lag - 1,
            "session": "overnight", "turnover_period": lag,
        }
        assert params.return_specs[intraday] == {
            "kind": "simple", "periods": 1, "lag": lag,
            "session": "intraday", "turnover_period": lag,
        }
    assert "return_columns" not in params.model_dump()
    assert "return_specs" not in params.model_dump()
    assert FactorAnalysisParams.model_validate_json(params.model_dump_json()) == params
    schema = FactorReportForm.model_json_schema()["properties"]["max_lag"]
    assert schema["type"] == "integer" and schema["minimum"] == 1 and schema["default"] == 1


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2", None, [1, 2]])
@pytest.mark.parametrize("model", [FactorAnalysisParams, FactorReportForm])
def test_max_lag_requires_a_strict_positive_integer(model, value):
    with pytest.raises(ValidationError):
        model(**DATES, columns=["score"], max_lag=value)


@pytest.mark.parametrize("model", [FactorAnalysisParams, FactorReportForm])
def test_old_holding_periods_are_not_reinterpreted_as_lags(model):
    with pytest.raises(ValidationError, match="return_periods"):
        model(**DATES, columns=["score"], return_periods=[1, 5, 20])


@pytest.mark.parametrize("column", ["r0o", "r1", "r1o", "r2", "time", "code", "date", "symbol", "circ_mv", "adjusted_open", "adjusted_close"])
def test_factor_columns_cannot_shadow_generated_or_join_columns(column):
    with pytest.raises(ValidationError, match="重名"):
        FactorAnalysisParams(**DATES, columns=[column], max_lag=2)
    form = FactorReportForm(**DATES, columns=[column], max_lag=2)
    with pytest.raises(ValidationError, match="重名"):
        form.build()


def test_lag_two_uses_three_trading_days_and_adj_factors_not_two_day_holding_return():
    prices = pd.DataFrame({
        "time": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]),
        "code": ["000001.SZ"] * 4,
        "open": [90.0, 55.0, 65.0, 70.0],
        "close": [100.0, 60.0, 66.0, 77.0],
        "adj_factor": [1.0, 2.0, 2.0, 2.0],
    })
    frame = evaluate_returns(prices.sample(frac=1, random_state=5), 2).reset_index(drop=True)
    assert frame.loc[0, ["r0o", "r1", "r1o", "r2"]].tolist() == pytest.approx([
        110 / 100 - 1, 120 / 110 - 1, 130 / 120 - 1, 132 / 130 - 1,
    ])
    assert frame.loc[1, ["r0o", "r1", "r1o", "r2"]].tolist() == pytest.approx([
        130 / 120 - 1, 132 / 130 - 1, 140 / 132 - 1, 154 / 140 - 1,
    ])
    assert frame.loc[2, ["r1o", "r2"]].isna().all()
    assert frame.loc[3, ["r0o", "r1", "r1o", "r2"]].isna().all()
    assert frame.loc[0, "r2"] != pytest.approx(132 / 100 - 1)
    # A missing factor observation on Monday must not change the price axis.
    sparse = frame[frame.time.isin(pd.to_datetime(["2025-01-03", "2025-01-07"]))]
    assert sparse.iloc[0].r1 == pytest.approx(120 / 110 - 1)


def test_missing_quotes_preserve_positions_and_independent_session_labels():
    dates = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"])
    prices = pd.DataFrame({
        "time": [*dates, *dates], "code": ["A"] * 3 + ["B"] * 3,
        "open": [9.0, 11.0, 13.0, 18.0, np.nan, 25.0],
        "close": [10.0, np.nan, 14.0, 20.0, np.nan, 30.0],
        "adj_factor": [1.0] * 6,
    })
    frame = evaluate_returns(prices, 2).set_index(["code", "time"])
    a = frame.loc[("A", dates[0])]
    assert a.r0o == pytest.approx(0.1)
    assert pd.isna(a.r1) and pd.isna(a.r1o)
    assert a.r2 == pytest.approx(14 / 13 - 1)
    b = frame.loc[("B", dates[0])]
    assert pd.isna(b.r0o) and pd.isna(b.r1) and pd.isna(b.r1o)
    assert b.r2 == pytest.approx(0.2)


def test_generated_graph_is_runtime_json_and_reads_only_open_close_and_adjustments():
    derivatives = _return_derivatives(3)
    query = FactorQuery.model_validate({
        "start_date": DATES["start"], "end_date": "2025-01-08", "codes": ["000001.SZ"],
        "factors": [], "derivatives": derivatives,
    })
    assert set(query.source_factors()) == {"open", "close", "adj_factor"}
    assert list(derivatives) == ["adjusted_open", "adjusted_close", "r0o", "r1", "r1o", "r2", "r2o", "r3"]


def test_factor_task_serializes_max_lag_but_not_derived_metadata(tmp_path):
    params = FactorAnalysisParams(**DATES, columns=["score"], max_lag=2)
    task = FactorTask.model_validate({
        "environment": {"lockfile": str(tmp_path / "uv.lock")}, "output": str(tmp_path / "output"),
        "factor": {"package": "factor-test", "version": "1.0.0", "wheel": "test.whl",
                   "sha256": "a" * 64, "entry": "factor_test:Factor"},
        "analysis": params.model_dump(mode="json"),
    })
    assert task.analysis.return_columns == ["r0o", "r1", "r1o", "r2"]
    assert task.model_dump()["analysis"]["max_lag"] == 2
    assert "return_specs" not in task.model_dump()["analysis"]
    assert "return_periods" not in task.model_dump()["analysis"]


def test_standalone_html_embeds_new_lag_specs_without_ipython(monkeypatch):
    module = import_module("scheme.report.html")
    monkeypatch.setattr(module, "_assets", lambda: {})
    report = SimpleNamespace(
        report_kind="factor", filenames={},
        parameters=FactorAnalysisParams(**DATES, columns=["score"], max_lag=2),
    )
    document = module.report_html(report)
    match = re.search(r'<script id="scheme-payload" type="application/json">(.*?)</script>', document, re.S)
    assert match is not None
    payload = json.loads(match.group(1))["report"]["parameters"]
    assert payload["return_columns"] == ["r0o", "r1", "r1o", "r2"]
    assert payload["return_specs"] == report.parameters.return_specs
    assert {spec["periods"] for spec in payload["return_specs"].values()} == {1}
