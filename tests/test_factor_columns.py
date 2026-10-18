"""Declared factor columns govern compute, preprocessing, saving and analysis."""

import sys
from abc import abstractmethod
from contextlib import nullcontext
from datetime import date
from inspect import isabstract, signature
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from scheme.base import Factor, FactorAnalysisParams, FactorParams, FactorReportForm
from scheme.base.projects.factor import algo
from scheme.execute.factor import api
from scheme.manage.parameters import defaults, inspect_project

DAY = date(2025, 1, 3)


def make_panel():
    index = pd.MultiIndex.from_product(
        [[DAY], ["000001.XSHE", "600000.XSHG"]], names=["date", "symbol"],
    )
    return pd.DataFrame({"score": [1.0, float("nan")], "auxiliary": ["up", None]}, index=index)


class DeclaredFactor(Factor[FactorParams]):
    columns = ["score", "auxiliary"]

    def compute(self, start: date, end: date) -> pd.DataFrame:
        """Return the supplied offline panel unchanged."""
        return self.panel


def make_factor(panel):
    factor = DeclaredFactor(FactorParams())
    factor.panel = panel
    return factor


def test_concrete_factor_requires_a_column_declaration_not_output_inference():
    with pytest.raises(TypeError, match="columns.*显式声明"):
        type("UndeclaredFactor", (Factor,), {"compute": lambda self, start, end: make_panel()})


@pytest.mark.parametrize("columns", [
    None, "score", [], (), {"score"}, [None], [True], [1], [""], ["  "], ["bad\0name"],
])
def test_invalid_class_column_declarations_fail_before_compute(columns):
    with pytest.raises(TypeError, match="columns.*有效字符串"):
        type("InvalidFactor", (Factor,), {
            "columns": columns, "compute": lambda self, start, end: make_panel(),
        })


@pytest.mark.parametrize("columns", [["score", "score"], ("score", "score")])
def test_class_column_declarations_must_be_unique(columns):
    with pytest.raises(ValueError, match="列不能重复"):
        type("DuplicateFactor", (Factor,), {
            "columns": columns, "compute": lambda self, start, end: make_panel(),
        })


@pytest.mark.parametrize("column", [
    "time", "code", "date", "symbol", "stock_pool_member", "circ_mv", "industry_l0",
])
def test_class_columns_cannot_shadow_keys_exposures_or_analysis_reserved_names(column):
    with pytest.raises(ValueError, match="保留列重名"):
        type("ReservedFactor", (Factor,), {
            "columns": [column], "compute": lambda self, start, end: make_panel(),
        })


def test_compute_wrapper_preserves_signature_metadata_and_abstract_behavior():
    class AbstractFactor(Factor[FactorParams]):
        @abstractmethod
        def compute(self, start: date, end: date) -> pd.DataFrame:
            """Abstract compute remains abstract."""
            ...

    assert isabstract(AbstractFactor)
    assert not hasattr(AbstractFactor.compute, "__wrapped__")
    with pytest.raises(TypeError, match="abstract"):
        AbstractFactor(FactorParams())

    class ConcreteFactor(AbstractFactor):
        columns = ("score", "auxiliary")
        compute = DeclaredFactor.compute.__wrapped__

    class InheritedFactor(ConcreteFactor):
        pass

    assert not isabstract(ConcreteFactor)
    assert InheritedFactor.columns is ConcreteFactor.columns
    assert InheritedFactor.compute is ConcreteFactor.compute
    wrapped = ConcreteFactor.compute
    original = wrapped.__wrapped__
    assert not hasattr(original, "__wrapped__")
    assert signature(wrapped) == signature(original)
    assert wrapped.__name__ == original.__name__
    assert wrapped.__doc__ == original.__doc__
    assert wrapped.__annotations__ == original.__annotations__
    factor = InheritedFactor(FactorParams())
    factor.panel = make_panel().assign(extra=1)
    with pytest.raises(ValueError, match="额外.*extra"):
        factor.compute(DAY, DAY)


@pytest.mark.parametrize("raw", [False, True])
def test_direct_compute_accepts_only_declared_columns_and_canonicalizes_raw_tables(raw):
    panel = make_panel().iloc[::-1, ::-1]
    data = panel.reset_index().rename(columns={"date": "time", "symbol": "code"}) if raw else panel
    actual = make_factor(data).compute(DAY, DAY)
    pd.testing.assert_frame_equal(actual, panel)
    if not raw:
        assert actual is panel


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("kind,detail", [
    ("missing", "缺少.*auxiliary"), ("extra", "额外.*extra"), ("duplicate", "列不能重复"),
])
def test_direct_compute_rejects_missing_extra_and_duplicate_factor_columns(raw, kind, detail):
    panel = make_panel()
    if raw:
        panel = panel.reset_index().rename(columns={"date": "time", "symbol": "code"})
    if kind == "missing":
        panel = panel.drop(columns="auxiliary")
    elif kind == "extra":
        panel = panel.assign(extra=1)
    else:
        panel = panel.rename(columns={"auxiliary": "score"})
    with pytest.raises(ValueError, match=detail):
        make_factor(panel).compute(DAY, DAY)


@pytest.mark.parametrize("kind,detail", [
    ("duplicate_key", "唯一"), ("null_key", "非空值"),
    ("string_date", "date.*日期"), ("intraday", "日历日期"),
])
def test_raw_compute_has_the_same_strict_key_and_calendar_date_checks(kind, detail):
    raw = make_panel().reset_index().rename(columns={"date": "time", "symbol": "code"})
    if kind == "duplicate_key":
        raw = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    elif kind == "null_key":
        raw.loc[0, "code"] = None
    elif kind == "string_date":
        raw["time"] = DAY.isoformat()
    else:
        raw["time"] = pd.Timestamp("2025-01-03 01:00:00")
    with pytest.raises(ValueError, match=detail):
        make_factor(raw).compute(DAY, DAY)


@pytest.mark.parametrize("specialized", [False, True])
def test_factor_report_forms_do_not_publish_or_accept_columns(specialized):
    class CustomAnalysis(FactorAnalysisParams):
        window: int = 20

    form = FactorReportForm[CustomAnalysis] if specialized else FactorReportForm
    assert "columns" not in form.model_fields
    assert "columns" not in form.model_json_schema()["properties"]
    assert "columns" not in defaults(form)
    assert form().build().columns == []
    with pytest.raises(ValidationError, match=r"columns.*[\s\S]*extra_forbidden"):
        form(columns=["score"])


def test_analysis_columns_default_to_independent_internal_lists():
    first = FactorAnalysisParams(start=DAY, end=DAY)
    second = FactorAnalysisParams(start=DAY, end=DAY)
    assert first.columns == second.columns == []
    assert first.columns is not second.columns


@pytest.mark.parametrize("data", ["offlineFactorTable", None])
def test_public_preprocess_uses_class_columns_without_a_columns_argument(monkeypatch, data):
    if data is None:
        data = make_panel()
    expected = make_panel()
    helper = Mock(return_value=expected)
    monkeypatch.setattr(algo, "preprocess", helper)
    factor = DeclaredFactor(FactorParams())
    session, settings = object(), object()
    assert "columns" not in signature(Factor.preprocess).parameters
    assert factor.preprocess(data, session=session, settings=settings) is expected
    helper.assert_called_once_with(
        data, session=session, columns=DeclaredFactor.columns,
        market_value_column="circ_mv", industry_column="industry_l0", settings=settings,
    )
    helper.reset_mock()
    with pytest.raises(TypeError, match="columns"):
        factor.preprocess(data, columns=["score"])
    helper.assert_not_called()


def test_params_run_binds_all_declared_columns_on_the_same_analysis_instance(monkeypatch):
    parameters = FactorAnalysisParams(start=DAY, end=DAY, columns=["score"])
    report, settings = object(), object()
    analyze = Mock(return_value=report)
    monkeypatch.setattr(api, "analyze_factors", analyze)
    assert parameters.run(DeclaredFactor, settings=settings) is report
    factor, bound = analyze.call_args.args
    assert isinstance(factor, DeclaredFactor)
    assert type(factor.params) is FactorParams
    assert bound is parameters
    assert parameters.columns == ["score", "auxiliary"]
    assert parameters.columns is not DeclaredFactor.columns
    parameters.columns.append("stale")
    assert DeclaredFactor.columns == ["score", "auxiliary"]
    parameters.run(DeclaredFactor, settings=settings)
    assert parameters.columns == DeclaredFactor.columns


@pytest.mark.parametrize("validate_factor", [False, True])
def test_inspect_project_automatically_saves_declared_columns(tmp_path, monkeypatch, validate_factor):
    module = ModuleType("factor_columns_fixture")
    module.__file__ = str(tmp_path / "src" / module.__name__ / "__init__.py")
    module.Factor, module.FactorReportForm = DeclaredFactor, FactorReportForm
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{module.__name__}"\n')
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    validate = Mock()
    monkeypatch.setattr(DeclaredFactor, "validate", validate)
    definition = inspect_project(tmp_path)
    assert "columns" not in definition["schemas"]["form"]["properties"]
    assert "columns" not in definition["values"]["form"]
    payload = inspect_project(
        tmp_path, {"form": {"start": DAY.isoformat(), "end": DAY.isoformat()}},
        validate_factor=validate_factor,
    )
    assert payload["analysis"]["columns"] == DeclaredFactor.columns
    assert "columns" not in payload["factor"]
    if validate_factor:
        validate.assert_called_once_with(DAY, DAY)
    else:
        validate.assert_not_called()


def test_direct_analyze_binds_all_columns_and_keeps_report_metadata(monkeypatch):
    panel = make_panel().assign(auxiliary=[2.0, 3.0])
    factor = SimpleNamespace(columns=("score", "auxiliary"), compute=Mock(return_value=panel))
    parameters = FactorAnalysisParams(start=DAY, end=DAY, columns=["score"])
    session = Mock()
    session.run.return_value = panel
    monkeypatch.setattr(api, "create_session", lambda settings: nullcontext(session))
    monkeypatch.setattr(api, "query", lambda *args, **kwargs: nullcontext(
        SimpleNamespace(data_ref="offlinePrices"),
    ))
    monkeypatch.setattr(api, "FactorAnalysisResult", lambda **kwargs: SimpleNamespace(**kwargs))
    report = api.analyze_factors(factor, parameters, settings=object())
    factor.compute.assert_called_once_with(DAY, DAY)
    assert parameters.columns == ["score", "auxiliary"]
    assert report.parameters.columns == parameters.columns
    assert report.parameters.columns is not parameters.columns
    uploaded = session.upload.call_args.args[0]
    assert uploaded["soloFactors"].tolist() == ["score", "auxiliary"]
    assert set(uploaded["soloFactorSource"].columns) == {"time", "code", "score", "auxiliary"}


def test_dynamic_return_name_conflicts_are_checked_when_binding_not_defining():
    class ReturnNamedFactor(Factor[FactorParams]):
        columns = ("P5L2",)

        def compute(self, start, end):
            raise AssertionError("conflicting factor must not compute")

    parameters = FactorAnalysisParams(start=DAY, end=DAY, period=5, max_lag=2)
    with pytest.raises(ValueError, match="收益列或分析保留列重名"):
        parameters.run(ReturnNamedFactor)
    assert parameters.columns == []
    with pytest.raises(ValueError, match="收益列或分析保留列重名"):
        api.analyze_factors(ReturnNamedFactor(FactorParams()), parameters)
    nonconflicting = FactorAnalysisParams(start=DAY, end=DAY)
    nonconflicting._bind_columns(ReturnNamedFactor)
    assert nonconflicting.columns == ["P5L2"]


@pytest.mark.parametrize("column", ["time", "circ_mv", "industry_l0", "stock_pool_member", "P1L1"])
def test_binding_retains_reserved_and_return_column_conflict_validation(column):
    parameters = FactorAnalysisParams(start=DAY, end=DAY)
    with pytest.raises(ValueError, match="收益列或分析保留列重名"):
        parameters._bind_columns(SimpleNamespace(columns=(column,)))
    assert parameters.columns == []


@pytest.mark.parametrize("raw", [False, True])
def test_prepare_panel_never_silently_drops_undeclared_columns(raw):
    panel = make_panel().assign(extra=1)
    if raw:
        panel = panel.reset_index().rename(columns={"date": "time", "symbol": "code"})
    parameters = FactorAnalysisParams(start=DAY, end=DAY, columns=["score", "auxiliary"])
    with pytest.raises(ValueError, match="额外.*extra"):
        api.prepare_panel(panel, parameters)
