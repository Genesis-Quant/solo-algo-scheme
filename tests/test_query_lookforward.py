from datetime import timedelta
from importlib import import_module
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError
from runtime.apps.query import FactorQuery as RuntimeFactorQuery
from scheme.base.projects.factor.preprocess import PREPROCESS_SCRIPT
from scheme.config import DolphinSettings
from scheme.data import FactorQuery

api = import_module("scheme.data.dolphindb.api")


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


def test_default_zero_and_runtime_schema_remains_unchanged():
    assert FactorQuery.model_validate(request()).lookforward == timedelta(0)
    assert "lookforward" in FactorQuery.model_json_schema()["properties"]
    assert "lookforward" not in RuntimeFactorQuery.model_fields
    with pytest.raises(ValidationError):
        RuntimeFactorQuery.model_validate(request(lookforward="P1D"))


def test_preserves_runtime_unknown_field_and_bool_checks():
    with pytest.raises(ValidationError):
        FactorQuery.model_validate(request(lookfoward="P1D"))
    with pytest.raises(ValidationError):
        FactorQuery.model_validate(
            request(
                derivatives={
                    "bad": {
                        "type": "CS",
                        "op": "unary.rank_pct",
                        "fields": {"col": "close"},
                        "on": "missing",
                    }
                }
            )
        )
    with pytest.raises(ValidationError):
        FactorQuery.model_validate(
            request(
                derivatives={
                    "number": {
                        "type": "DIRECT",
                        "op": "unary.get",
                        "fields": {"col": "close"},
                    }
                },
                filters=["number"],
            )
        )


def query_parameters(monkeypatch, value):
    monkeypatch.setattr(api, "QueryResult", lambda parameters, **_: parameters)
    return api.query(value, settings=settings())


def test_calculation_and_output_bounds_are_separate(monkeypatch):
    parameters = query_parameters(monkeypatch, request(lookback="P110D", lookforward="P20D"))
    assert parameters["start"] == pd.Timestamp("2026-01-10")
    assert parameters["end"] == pd.Timestamp("2026-05-29")
    assert parameters["outputStart"] == pd.Timestamp("2026-04-30")
    assert parameters["outputEnd"] == pd.Timestamp("2026-05-09")


@pytest.mark.parametrize("as_model", [False, True])
def test_original_runtime_model_defaults_to_no_lookforward(monkeypatch, as_model):
    value = RuntimeFactorQuery.model_validate(request()) if as_model else request()
    parameters = query_parameters(monkeypatch, value)
    assert parameters["end"] == parameters["outputEnd"] == pd.Timestamp("2026-05-09")
    assert parameters["start"] == parameters["outputStart"]


def test_scheme_model_keeps_lookforward(monkeypatch):
    model = FactorQuery.model_validate(request(lookforward="P20D"))
    parameters = query_parameters(monkeypatch, model)
    assert parameters["end"] == pd.Timestamp("2026-05-29")
    assert parameters["outputEnd"] == pd.Timestamp("2026-05-09")


def test_daily_axis_normalizes_fractional_duration(monkeypatch):
    parameters = query_parameters(monkeypatch, request(lookforward="P1DT12H"))
    assert parameters["end"] == pd.Timestamp("2026-05-10")


def test_negative_shift_dependency_is_loaded_without_becoming_an_output(monkeypatch):
    parameters = query_parameters(
        monkeypatch,
        request(
            derivatives={
                "next_close": {
                    "type": "TS",
                    "op": "unary.shift",
                    "fields": {"col": "close"},
                    "params": {"periods": -1},
                }
            },
            lookforward="P20D",
        ),
    )
    assert list(parameters["columns"]) == ["time", "code", "close", "next_close"]
    assert list(parameters["factors"]) == ["close"]
    assert parameters["codes"].tolist() == ["600000.SH"]


def test_output_calendar_and_rows_use_original_end():
    assert (
        "project_factor_output(filtered, request.columns, request.outputStart, request.outputEnd)"
        in api.QUERY_SCRIPT
    )
    assert "calendar >= request.outputStart && calendar < request.outputEnd" in api.QUERY_SCRIPT
    assert 'request["outputEnd"] = request["end"]' in PREPROCESS_SCRIPT


def test_borrowed_session_is_cleaned_without_closing(monkeypatch):
    session = Mock()
    monkeypatch.setattr(api.query_api, "redirect_session_output", lambda _: None)
    monkeypatch.setattr(api, "upload", lambda *args: None)
    release = Mock()
    monkeypatch.setattr(api, "release", release)
    with api.query(request(lookforward="P20D"), session=session, settings=settings()) as result:
        assert result.session is session
    session.close.assert_not_called()
    assert result.closed
    assert release.call_count == 2
