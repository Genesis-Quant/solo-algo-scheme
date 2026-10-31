from datetime import timedelta
from importlib import import_module
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError
from query import FactorQuery as QueryFactorQuery
from query.apps.query.api import query_parameters

from scheme.base.projects.factor.preprocess import PREPROCESS_SCRIPT
from scheme.config import DolphinSettings
from scheme.data import FactorQuery

api = import_module("scheme.data.dolphindb.api")
query_api = import_module("query.apps.query.api")


def request(**overrides):
    return {
        "start_date": "2026-04-30",
        "end_date": "2026-05-08",
        "codes": ["600000.SH"],
        "factors": ["close"],
        **overrides,
    }


def settings():
    return DolphinSettings(username="unit-test", password="unused")


@pytest.mark.parametrize(
    "value, expected",
    [
        (timedelta(0), timedelta(0)),
        (timedelta(days=20), timedelta(days=20)),
        ("P20D", timedelta(days=20)),
        ("PT12H", timedelta(hours=12)),
    ],
)
def test_accepts_nonnegative_duration(value, expected):
    model = FactorQuery.model_validate(request(lookforward=value))
    assert model.lookforward == expected
    restored = FactorQuery.model_validate_json(model.model_dump_json())
    assert restored.lookforward == expected


@pytest.mark.parametrize(
    "value", [True, False, 1, 0, 1.5, None, [], {}, "bad", "-P1D", timedelta(days=-1)]
)
def test_rejects_undeclared_or_negative_duration(value):
    with pytest.raises(ValidationError):
        FactorQuery.model_validate(request(lookforward=value))


def test_default_zero_and_shared_query_schema():
    assert FactorQuery is QueryFactorQuery
    assert FactorQuery.model_validate(request()).lookforward == timedelta(0)
    assert "lookforward" in FactorQuery.model_json_schema()["properties"]


def test_preserves_unknown_field_and_bool_checks():
    with pytest.raises(ValidationError):
        FactorQuery.model_validate(request(lookfoward="P1D"))
    with pytest.raises(ValidationError):
        FactorQuery.model_validate(
            request(derivatives={
                "bad": {"type": "CS", "op": "unary.rank_pct", "fields": {"col": "close"}, "on": "missing"}
            })
        )
    with pytest.raises(ValidationError):
        FactorQuery.model_validate(
            request(derivatives={
                "number": {"type": "DIRECT", "op": "unary.get", "fields": {"col": "close"}}
            }, filters=["number"])
        )


def test_calculation_and_output_bounds_are_separate():
    parameters = query_parameters(request(lookback="P110D", lookforward="P20D"), settings())
    assert parameters["start"] == pd.Timestamp("2026-01-10")
    assert parameters["end"] == pd.Timestamp("2026-05-29")
    assert parameters["outputStart"] == pd.Timestamp("2026-04-30")
    assert parameters["outputEnd"] == pd.Timestamp("2026-05-09")


@pytest.mark.parametrize("as_model", [False, True])
def test_model_and_dict_default_to_no_lookforward(as_model):
    value = FactorQuery.model_validate(request()) if as_model else request()
    parameters = query_parameters(value, settings())
    assert parameters["end"] == parameters["outputEnd"] == pd.Timestamp("2026-05-09")
    assert parameters["start"] == parameters["outputStart"]


def test_daily_axis_normalizes_fractional_duration():
    parameters = query_parameters(request(lookforward="P1DT12H"), settings())
    assert parameters["end"] == pd.Timestamp("2026-05-10")


def test_negative_shift_dependency_is_loaded_without_becoming_an_output():
    parameters = query_parameters(request(
        factors=[],
        derivatives={"next_close": {
            "type": "TS", "op": "unary.shift", "fields": {"col": "close"}, "params": {"periods": -1}
        }},
        lookforward="P20D",
    ), settings())
    assert list(parameters["columns"]) == ["time", "code", "next_close"]
    assert list(parameters["factors"]) == ["close"]
    assert parameters["codes"].tolist() == ["600000.SH"]


def test_output_calendar_and_rows_use_original_end():
    assert "project_factor_output(filtered, request.columns, request.outputStart, request.outputEnd)" in api.QUERY_SCRIPT
    assert "calendar >= request.outputStart && calendar < request.outputEnd" in api.QUERY_SCRIPT
    assert 'request["outputEnd"] = request["end"]' in PREPROCESS_SCRIPT


def test_borrowed_session_is_cleaned_without_closing(monkeypatch):
    session = Mock()
    monkeypatch.setattr(query_api, "redirect_session_output", lambda _: None)
    monkeypatch.setattr(query_api, "initialize_session", lambda _: None)
    with api.query(request(lookforward="P20D"), session=session, settings=settings()) as result:
        assert result.session is session
    session.close.assert_not_called()
    assert result.closed
    scripts = "\n".join(call.args[0] for call in session.run.call_args_list)
    assert 'undef("coreQueryData", VAR)' in scripts
    assert 'undef("coreQueryCalendar", VAR)' in scripts
    assert 'undef("buildQuery", DEF)' not in scripts


def test_no_arena_runtime_is_needed():
    import importlib.metadata as metadata

    with pytest.raises(metadata.PackageNotFoundError):
        metadata.version("arena-runtime")
    assert metadata.version("query") == "0.1.0"
