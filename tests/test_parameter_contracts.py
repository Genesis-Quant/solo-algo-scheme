"""Focused parameter/form/projection contracts, without research services or output."""

from dataclasses import fields, is_dataclass
from datetime import date, timedelta
from typing import Self, assert_type, get_args, get_origin, get_type_hints
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, ValidationError

from scheme import Factor, ResearchContext, StockPool, Strategy, Universe
from scheme.base import (
    Algo,
    ControlAnalysisParams,
    ControlParams,
    ControlReportForm,
    ExecutionAnalysisParams,
    ExecutionParams,
    ExecutionReportForm,
    FactorAnalysisParams,
    FactorParams,
    FactorReportForm,
    ModelAlgo,
    ModelAnalysisParams,
    ModelParams,
    ModelReportForm,
    OptimizeAnalysisParams,
    OptimizeParams,
    OptimizeReportForm,
    ReportForm,
    StrategyAnalysisParams,
    StrategyParams,
)
from scheme.execute.factor.result import FactorAnalysisResult
from scheme.execute.strategy.assembly import (
    context_type,
    parameter_type,
    project_parameters,
    validate_context,
)
from scheme.execute.strategy.result import BacktestResult
from scheme.manage.parameters import defaults

DATES = {"start": "2025-01-01", "end": "2025-01-03"}
STAGES = (
    (ModelParams, ModelAnalysisParams, ModelReportForm, {}),
    (OptimizeParams, OptimizeAnalysisParams, OptimizeReportForm, {"model": "model_test:Model"}),
    (ControlParams, ControlAnalysisParams, ControlReportForm,
     {"model": "model_test:Model", "optimize": "optimize_test:Optimize"}),
    (ExecutionParams, ExecutionAnalysisParams, ExecutionReportForm,
     {"model": "model_test:Model", "optimize": "optimize_test:Optimize",
      "control": "control_test:Control"}),
)
FORMS = ((FactorParams, FactorAnalysisParams, FactorReportForm, {"columns": ["f"]}), *STAGES)
PARAMETERS = (
    (FactorParams, {}), (FactorAnalysisParams, {"columns": ["f"]}),
    (StrategyParams, {}), (StrategyAnalysisParams, {}),
    *((analysis, selectors) for _, analysis, _, selectors in STAGES),
)


class WindowParams(FactorParams):
    window: int = Field(gt=0)


class WindowAnalysis(WindowParams, FactorAnalysisParams):
    pass


class WindowForm(ReportForm[WindowAnalysis]):
    start: date
    end: date
    window: int
    pool: StockPool = StockPool.CSI300

    def build(self) -> WindowAnalysis:
        return WindowAnalysis(
            start=self.start, end=self.end, window=self.window,
            universe=Universe(pool=self.pool), columns=["momentum"],
        )


class WindowFactor(Factor[WindowParams]):
    def compute(self, start: date, end: date) -> pd.DataFrame:
        raise AssertionError("contract tests must not compute or query")


class SelectionParams(ModelParams):
    n_select: int = Field(ge=1)


class SelectionAnalysis(SelectionParams, ModelAnalysisParams):
    pass


class SelectionForm(ReportForm[SelectionAnalysis]):
    start: date
    end: date
    n_select: int
    cash: float = Field(default=100_000, gt=0)

    def build(self) -> SelectionAnalysis:
        return SelectionAnalysis(
            start=self.start, end=self.end, n_select=self.n_select,
            config={"cash": self.cash},
        )


class ContractState(ResearchContext[float]):
    pass


class SelectionModel(ModelAlgo[SelectionParams, ContractState]):
    pass


@pytest.mark.parametrize("model,values", PARAMETERS)
def test_runtime_dates_are_required_validated_and_round_trip(model, values):
    schema = model.model_json_schema()
    assert {"start", "end"} <= set(schema["required"])
    for name in ("start", "end"):
        assert schema["properties"][name]["format"] == "date"
    params = model.model_validate({**DATES, **values})
    assert (params.start, params.end) == (date(2025, 1, 1), date(2025, 1, 3))
    assert model.model_validate_json(params.model_dump_json()) == params
    for missing in ("start", "end"):
        with pytest.raises(ValidationError) as error:
            model.model_validate({k: v for k, v in {**DATES, **values}.items() if k != missing})
        assert any(item["loc"] == (missing,) and item["type"] == "missing"
                   for item in error.value.errors())
    for end in ("2025-01-01", "2024-12-31"):
        with pytest.raises(ValidationError, match="start < end"):
            model.model_validate({**DATES, **values, "end": end})
    with pytest.raises(ValidationError, match="start"):
        model.model_validate({**DATES, **values, "start": "not-a-date"})


def test_factor_and_strategy_runtime_models_are_independent_direct_models():
    assert FactorParams.__bases__ == (BaseModel,)
    assert StrategyParams.__bases__ == (BaseModel,)
    assert FactorAnalysisParams.__bases__ == (FactorParams,)
    assert StrategyAnalysisParams.__bases__ == (StrategyParams,)
    assert set(FactorParams.model_fields) == {"start", "end", "universe"}
    assert set(StrategyParams.model_fields) == {"start", "end", "universe"}
    assert set(FactorParams.__annotations__) == set(FactorParams.model_fields)
    assert set(StrategyParams.__annotations__) == set(StrategyParams.model_fields)
    assert not issubclass(FactorParams, StrategyParams)
    assert not issubclass(StrategyParams, FactorParams)


@pytest.mark.parametrize("runtime,analysis,form,selectors", STAGES)
def test_stage_params_are_algorithm_only_and_analysis_declares_own_runtime(
    runtime, analysis, form, selectors,
):
    assert runtime.__bases__ == (BaseModel,)
    assert runtime.model_fields == {}
    assert analysis.__bases__ == (runtime,)
    assert analysis.__mro__ == (analysis, runtime, BaseModel, object)
    assert set(analysis.model_fields) == set(analysis.__annotations__)
    assert set(StrategyAnalysisParams.model_fields) <= set(analysis.model_fields)
    assert set(analysis.model_fields) - set(StrategyAnalysisParams.model_fields) == {
        name for name in ("model", "optimize", "control", "execution")
        if name != runtime.__name__.removesuffix("Params").lower()
    }
    assert not issubclass(analysis, StrategyParams)
    assert analysis.model_validate({**DATES, **selectors}).start == date(2025, 1, 1)


@pytest.mark.parametrize("runtime,analysis,form,values", FORMS)
def test_shared_form_preserves_ui_and_concrete_analysis_contract(
    runtime, analysis, form, values,
):
    assert issubclass(form, ReportForm)
    assert ReportForm.model_fields == {}
    assert form.analysis_model() is analysis
    assert not issubclass(form, runtime)
    assert not issubclass(form, analysis)
    schema = form.model_json_schema()
    assert set(schema["properties"]) == set(form.model_fields)
    assert not {"start", "end"} & set(schema.get("required", []))
    assert schema["properties"]["start"]["default"] == "2020-01-01"
    assert schema["properties"]["end"]["default"] == "2027-01-01"
    assert schema["additionalProperties"] is False
    assert {"start", "end", "pool", "lookback"} <= set(schema["properties"])
    assert not {"universe", "config", "symbols"} & set(schema["properties"])
    assert all(not field.get("x-hidden") for field in schema["properties"].values())
    built = form.model_validate({**DATES, **values}).build()
    assert type(built) is analysis
    for hidden in ("universe", "config", "undeclared"):
        with pytest.raises(ValidationError) as error:
            form.model_validate({**DATES, **values, hidden: {}})
        assert any(item["loc"] == (hidden,) and item["type"] == "extra_forbidden"
                   for item in error.value.errors())


@pytest.mark.parametrize("runtime,analysis,form,values", FORMS)
def test_form_defaults_are_exposed_as_json_and_build_without_dates(
    runtime, analysis, form, values,
):
    initial = defaults(form)
    assert initial["start"] == "2020-01-01"
    assert initial["end"] == "2027-01-01"
    assert initial["pool"] == StockPool.CSI300
    assert initial["lookback"] == "PT0S"
    built = form.model_validate({**initial, **values}).build()
    assert type(built) is analysis
    assert (built.start, built.end) == (date(2020, 1, 1), date(2027, 1, 1))
    assert built.universe.pool == StockPool.CSI300
    assert built.universe.lookback == timedelta(0)
    overridden = form.model_validate({**initial, **values, **DATES}).build()
    assert (overridden.start, overridden.end) == (date(2025, 1, 1), date(2025, 1, 3))


def test_factor_form_defaults_match_web_research_settings():
    initial = defaults(FactorReportForm)
    assert initial == {
        "start": "2020-01-01", "end": "2027-01-01", "pool": StockPool.CSI300,
        "lookback": "PT0S", "return_periods": [1, 5, 20], "groups": 5,
        "n_select": 10, "weight": "market_value", "calendar_symbol": "000300.XSHG",
    }
    assert FactorReportForm.model_fields["columns"].is_required()
    built = FactorReportForm(columns=["momentum"]).build()
    assert built.return_periods == [1, 5, 20]
    assert built.groups == 5
    assert built.n_select == 10
    assert built.weight == "market_value"
    assert built.calendar_symbol == "000300.XSHG"
    first = FactorReportForm(columns=["momentum"])
    first.return_periods.append(60)
    assert FactorReportForm(columns=["momentum"]).return_periods == [1, 5, 20]


@pytest.mark.parametrize("runtime,analysis,form,values", FORMS)
def test_form_cross_field_dates_are_validated_by_build_not_constructor(
    runtime, analysis, form, values,
):
    for end in ("2025-01-01", "2024-12-31"):
        pending = form.model_validate({**DATES, **values, "end": end})
        assert pending.end == date.fromisoformat(end)
        with pytest.raises(ValidationError, match="start < end"):
            pending.build()


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"columns": ["f", "f"]}, "因子列不能重复"),
        ({"return_periods": [1, 1]}, "收益持有期不能重复"),
        ({"return_periods": [0]}, "正整数"),
    ],
)
def test_factor_form_analysis_cross_field_validation_is_deferred(updates, message):
    form = FactorReportForm.model_validate({**DATES, "columns": ["f"], **updates})
    with pytest.raises(ValidationError, match=message):
        form.build()


@pytest.mark.parametrize("runtime,analysis,form,selectors", STAGES)
def test_stage_form_selectors_and_engine_values_have_explicit_wire_shape(
    runtime, analysis, form, selectors,
):
    built = form.model_validate({
        **DATES, **selectors, "cash": 350_000, "commission": 0.001, "tax": 0.002,
        "benchmark": "000300.SH", "pool": StockPool.CSI500, "lookback": "P5D",
    }).build()
    assert built.config == {"cash": 350_000, "commission": 0.001, "tax": 0.002}
    assert built.benchmark == "000300.XSHG"
    assert built.universe.pool == StockPool.CSI500
    assert built.universe.lookback == timedelta(days=5)
    assert not {"cash", "commission", "tax", "pool", "lookback"} & set(built.model_dump())
    schema = form.model_json_schema()
    for name in selectors:
        assert name in schema["required"]
        assert schema["properties"][name]["x-algo-kind"] == name
    current = runtime.__name__.removesuffix("Params").lower()
    assert current not in form.model_fields
    defaults = {"optimize": "risk_parity", "control": "no_control", "execution": "direct_execution"}
    for name in set(defaults) - set(selectors) - {current}:
        assert schema["properties"][name]["const"] == defaults[name]
    assert analysis.model_validate_json(built.model_dump_json()) == built


@pytest.mark.parametrize("runtime,analysis,form,selectors", STAGES)
def test_stage_extras_are_ignored_at_runtime_but_preserved_for_analysis(
    runtime, analysis, form, selectors,
):
    assert runtime.model_config["extra"] == "ignore"
    assert analysis.model_config["extra"] == "allow"
    assert runtime.model_validate({**DATES, "custom_window": 20}).model_dump() == {}
    params = analysis.model_validate({**DATES, **selectors, "custom_window": 20})
    assert params.model_extra == {"custom_window": 20}
    restored = analysis.model_validate_json(params.model_dump_json())
    assert restored.model_extra == {"custom_window": 20}
    assert params.model_json_schema()["additionalProperties"] is True


@pytest.mark.parametrize("model,values", [(FactorParams, {}), (FactorAnalysisParams, {"columns": ["f"]})])
def test_factor_params_forbid_undeclared_fields(model, values):
    assert model.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError) as error:
        model.model_validate({**DATES, **values, "window": 20})
    assert any(item["loc"] == ("window",) and item["type"] == "extra_forbidden"
               for item in error.value.errors())


@pytest.mark.parametrize("model", [StrategyParams, StrategyAnalysisParams])
def test_strategy_params_preserve_algorithm_extras(model):
    params = model.model_validate({**DATES, "window": 20})
    assert model.model_config["extra"] == "allow"
    assert params.model_extra == {"window": 20}
    assert model.model_validate_json(params.model_dump_json()).model_extra == {"window": 20}


@pytest.mark.parametrize("form_type,values", [(WindowForm, {"window": 7}), (SelectionForm, {"n_select": 3})])
def test_custom_form_schema_does_not_inherit_runtime_or_analysis_fields(form_type, values):
    schema = form_type.model_json_schema()
    assert set(schema["properties"]) == set(form_type.__annotations__)
    assert not {"universe", "config", "columns", "groups", "optimize", "control", "execution"} & set(
        schema["properties"]
    )
    assert form_type.model_validate({**DATES, **values}).build()


def test_direct_custom_factor_form_builds_exact_analysis_and_projects_runtime(monkeypatch):
    form = WindowForm(**DATES, window=7)
    analysis = form.build()
    assert_type(analysis, WindowAnalysis)
    assert type(analysis) is WindowAnalysis
    assert isinstance(analysis, WindowParams)
    assert WindowAnalysis.model_validate_json(analysis.model_dump_json()) == analysis
    report = FactorAnalysisResult(
        **{name: pd.DataFrame() for name in FactorAnalysisResult.filenames},
        parameters=analysis.model_copy(deep=True),
    )
    analyze = Mock(return_value=report)
    monkeypatch.setattr("scheme.execute.factor.api.analyze_factors", analyze)
    result = analysis.run(WindowFactor)
    assert_type(result, FactorAnalysisResult[WindowAnalysis])
    assert result is report
    factor, received = analyze.call_args.args
    assert received is analysis
    assert type(factor.params) is WindowParams
    assert factor.params.window == 7
    assert set(factor.params.model_dump()) == {"start", "end", "universe", "window"}
    assert type(result.parameters) is WindowAnalysis
    assert result.parameters == analysis
    analysis.columns.append("changed")
    assert result.parameters.columns == ["momentum"]


def test_direct_custom_model_form_runs_with_exact_analysis_type(monkeypatch):
    analysis = SelectionForm(**DATES, n_select=4, cash=25_000).build()
    assert_type(analysis, SelectionAnalysis)
    assert type(analysis) is SelectionAnalysis
    report = BacktestResult(**{name: pd.DataFrame() for name in BacktestResult.filenames})
    run = Mock(return_value=report)
    monkeypatch.setattr("scheme.execute.strategy.api.run_backtest", run)
    result = analysis.run(SelectionModel)
    assert_type(result, BacktestResult[SelectionAnalysis])
    assert result is report
    assert type(result.parameters) is SelectionAnalysis
    assert result.parameters == analysis
    assert result.parameters is not analysis
    algos, ctx, backtest = run.call_args.args
    assert type(algos[0]) is SelectionModel
    assert type(algos[0].params) is SelectionParams
    assert algos[0].params.model_dump() == {"n_select": 4}
    assert type(ctx) is ContractState
    assert type(backtest) is StrategyAnalysisParams
    assert backtest.config["cash"] == 25_000


@pytest.mark.parametrize(
    "form_type,field", [(WindowForm, "window"), (SelectionForm, "n_select")],
)
def test_custom_required_fields_and_build_validation(form_type, field):
    with pytest.raises(ValidationError, match=field):
        form_type.model_validate(DATES)
    with pytest.raises(ValidationError, match=field):
        form_type.model_validate({**DATES, field: "invalid"})
    pending = form_type.model_validate({**DATES, field: 0})
    with pytest.raises(ValidationError, match=field):
        pending.build()


def test_factor_run_revalidates_missing_or_invalid_concrete_runtime(monkeypatch):
    analyze = Mock()
    monkeypatch.setattr("scheme.execute.factor.api.analyze_factors", analyze)
    with pytest.raises(ValidationError, match="window"):
        FactorAnalysisParams(**DATES, columns=["f"]).run(WindowFactor)
    invalid = WindowAnalysis.model_construct(
        **FactorAnalysisParams(**DATES, columns=["f"]).model_dump(), window=0,
    )
    with pytest.raises(ValidationError, match="window"):
        invalid.run(WindowFactor)
    analyze.assert_not_called()


def test_projection_reconstructs_declared_model_instead_of_leaking_subclass():
    analysis = WindowForm(**DATES, window=11).build()
    # Pydantic's subtype fast path is why passing the model instance is not projection.
    assert WindowParams.model_validate(analysis) is analysis
    runtime = project_parameters(WindowParams, analysis)
    assert_type(runtime, WindowParams)
    assert type(runtime) is WindowParams
    assert runtime is not analysis
    assert runtime.model_dump() == {
        key: value for key, value in analysis.model_dump().items() if key in WindowParams.model_fields
    }
    assert not hasattr(runtime, "columns")
    base = project_parameters(FactorParams, analysis)
    assert_type(base, FactorParams)
    assert type(base) is FactorParams
    assert set(base.model_dump()) == {"start", "end", "universe"}


def test_projection_excludes_analysis_extras_even_for_allow_runtime_model():
    class Runtime(ModelParams):
        model_config = ConfigDict(extra="allow")
        count: int = Field(ge=1)

    class Analysis(Runtime, ModelAnalysisParams):
        pass

    class Model(ModelAlgo[Runtime, ContractState]):
        pass

    params = Analysis(**DATES, count=2, private_note="analysis only")
    assert Runtime.model_validate(params) is params
    projected = project_parameters(Runtime, params)
    assert type(projected) is Runtime
    assert projected.model_dump() == {"count": 2}
    assert projected.model_extra == {}
    strategy = Strategy(ContractState(), params=StrategyParams(**params.model_dump()), model=Model)
    assert type(strategy.algos[0].params) is Runtime
    assert strategy.algos[0].params.model_dump() == {"count": 2}
    with pytest.raises(ValidationError, match="count"):
        project_parameters(Runtime, StrategyParams(**DATES))
    with pytest.raises(ValidationError, match="count"):
        project_parameters(Runtime, StrategyParams(**DATES, count=0))


def test_projection_reads_declared_runtime_fields_from_analysis_extras():
    params = ModelAnalysisParams(**DATES, n_select="6", unrelated=1)
    projected = project_parameters(SelectionParams, params)
    assert_type(projected, SelectionParams)
    assert type(projected) is SelectionParams
    assert projected.model_dump() == {"n_select": 6}


@pytest.mark.parametrize("alias,inputs", [
    ("n_select", {"n_select": 17}),
    (AliasChoices("n_select", "selection"), {"selection": 17}),
    (AliasPath("config", "counts", -1), {"config": {"counts": [3, 17]}}),
    (AliasChoices(AliasPath("config", "count"), "n_select"), {"n_select": 17}),
])
def test_projection_aliases_preserve_canonical_and_legacy_semantics(alias, inputs):
    class Runtime(ModelParams):
        model_config = ConfigDict(extra="allow")
        count: int = Field(validation_alias=alias, gt=0)

    class Analysis(Runtime, ModelAnalysisParams):
        pass

    legacy = StrategyAnalysisParams(**DATES, count=99, unrelated=True, **inputs)
    expected = Runtime.model_validate(legacy, from_attributes=True)
    projected = project_parameters(Runtime, legacy)
    assert type(projected) is Runtime
    assert projected.count == expected.count == 17
    assert projected.model_extra == {}
    for source in (StrategyAnalysisParams(**DATES, count=17), {**DATES, "count": 17}):
        projected = project_parameters(Runtime, source)
        assert type(projected) is Runtime
        assert projected.count == 17
        assert projected.model_extra == {}

    canonical = Analysis(**DATES, **inputs)
    canonical.count = 23
    projected = project_parameters(Runtime, canonical)
    assert type(projected) is Runtime
    assert projected is not canonical
    assert projected.count == 23
    assert projected.model_extra == {}
    if "config" in inputs:
        assert canonical.config == inputs["config"]
    canonical.count = 0
    with pytest.raises(ValidationError):
        project_parameters(Runtime, canonical)
    with pytest.raises(ValidationError):
        project_parameters(Runtime, StrategyAnalysisParams(**DATES))


def test_projection_alias_preserves_legacy_model_analysis_extras():
    class Runtime(ModelParams):
        count: int = Field(alias="n_select", gt=0)

    analysis = ModelAnalysisParams(**DATES, n_select=17, private_note="research only")
    projected = project_parameters(Runtime, analysis)
    assert type(projected) is Runtime
    assert projected.model_dump() == {"count": 17}
    assert analysis.model_extra == {"n_select": 17, "private_note": "research only"}


def test_projection_preserves_nested_subtype_and_excluded_runtime_field():
    class Nested(BaseModel):
        value: int

    class NestedChild(Nested):
        detail: str

    class Runtime(ModelParams):
        nested: Nested
        hidden: int = Field(alias="secret", exclude=True)

    class Analysis(Runtime, ModelAnalysisParams):
        pass

    nested = NestedChild(value=3, detail="must survive projection")
    analysis = Analysis(**DATES, nested=nested, secret=17)
    projected = project_parameters(Runtime, analysis)
    assert type(projected) is Runtime
    assert projected.nested is nested
    assert type(projected.nested) is NestedChild
    assert projected.hidden == 17
    assert projected.model_dump() == {"nested": {"value": 3}}


def test_projection_alias_defaults_and_present_invalid_values():
    class Runtime(ModelParams):
        count: int = Field(default=20, alias="n_select", gt=0)
        labels: list[str] = Field(default_factory=list, validation_alias=AliasPath("config", "labels"))

    first = project_parameters(Runtime, ModelAnalysisParams(**DATES))
    second = project_parameters(Runtime, ModelAnalysisParams(**DATES))
    assert type(first) is Runtime
    assert first.count == second.count == 20
    assert first.labels == second.labels == []
    assert first.labels is not second.labels
    for invalid in (None, "invalid", 0):
        with pytest.raises(ValidationError):
            project_parameters(Runtime, ModelAnalysisParams(**DATES, count=17, n_select=invalid))


def test_pep695_multilevel_factor_parameter_resolution():
    class Generic[P: FactorParams](Factor[P]):
        def compute(self, start: date, end: date) -> pd.DataFrame:
            raise AssertionError("must not compute")

    class Forwarded[P: FactorParams](Generic[P]):
        pass

    class Concrete(Forwarded[WindowParams]):
        pass

    class Leaf(Concrete):
        pass

    assert parameter_type(Generic) is FactorParams
    assert parameter_type(Concrete) is WindowParams
    assert parameter_type(Leaf) is WindowParams
    assert_type(parameter_type(Leaf), type[WindowParams])


def test_pep695_multilevel_reordered_typevars_resolve_params_and_context():
    class Generic[P: BaseModel, C: ResearchContext[float]](Algo[P, C]):
        pass

    class Reordered[C: ResearchContext[float], P: BaseModel](Generic[P, C]):
        pass

    class Forwarded[P: BaseModel, C: ResearchContext[float]](Reordered[C, P]):
        pass

    class Partial[C: ResearchContext[float]](Forwarded[SelectionParams, C]):
        pass

    class Concrete(Partial[ContractState]):
        pass

    class Leaf(Concrete):
        pass

    for cls in (Concrete, Leaf):
        assert parameter_type(cls) is SelectionParams
        assert context_type(cls) is ContractState
        algo = cls(SelectionParams(n_select=2))
        validate_context(algo, ContractState())
        with pytest.raises(TypeError, match="ContractState"):
            validate_context(algo, ResearchContext[str]())
    assert_type(parameter_type(Leaf), type[SelectionParams])
    assert_type(context_type(Leaf), type[ContractState])


def test_generic_resolution_rejects_missing_model_and_nonresearch_context():
    class InvalidParameter(Algo[int, ContractState]):
        pass

    class Unrelated(BaseModel):
        pass

    class InvalidContext(Algo[SelectionParams, Unrelated]):
        pass

    for cls in (object, Unrelated, InvalidParameter):
        with pytest.raises(TypeError, match="BaseModel 参数泛型"):
            parameter_type(cls)
    with pytest.raises(TypeError, match="ResearchContext"):
        context_type(InvalidContext)


@pytest.mark.parametrize(
    "analysis,result_model",
    [
        (FactorAnalysisParams, FactorAnalysisResult),
        (StrategyAnalysisParams, BacktestResult),
        *((analysis, BacktestResult) for _, analysis, _, _ in STAGES),
    ],
)
def test_run_annotation_preserves_self_and_generic_result(analysis, result_model):
    run = analysis.run
    namespace = {
        **run.__globals__, "Factor": Factor, "Strategy": Strategy,
        "BacktestResult": BacktestResult, "FactorAnalysisResult": FactorAnalysisResult,
    }
    from scheme.base import ControlAlgo, ExecutionAlgo, OptimizeAlgo
    from scheme.config import DolphinSettings

    namespace.update(
        ModelAlgo=ModelAlgo, OptimizeAlgo=OptimizeAlgo, ControlAlgo=ControlAlgo,
        ExecutionAlgo=ExecutionAlgo, DolphinSettings=DolphinSettings,
    )
    bindings = {param.__name__: param for param in run.__type_params__}
    hints = get_type_hints(run, globalns=namespace, localns=bindings)
    assert get_origin(hints["return"]) is result_model
    assert get_args(hints["return"]) == (Self,)
    assert run.__type_params__


@pytest.mark.parametrize("result_model", [FactorAnalysisResult, BacktestResult])
def test_result_dataclasses_expose_generic_parameter_field(result_model):
    assert is_dataclass(result_model)
    assert len(result_model.__type_params__) == 1
    assert "parameters" in {field.name for field in fields(result_model)}
    assert "parameters" not in result_model.filenames
    hints = get_type_hints(result_model)
    assert get_args(hints["parameters"]) == (result_model.__type_params__[0], type(None))
    result = result_model(**{name: pd.DataFrame() for name in result_model.filenames})
    assert result.parameters is None
