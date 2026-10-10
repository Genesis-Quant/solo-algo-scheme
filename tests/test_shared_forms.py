"""公共表单绑定具体研究参数；不执行研究或访问数据服务。"""

import json
import sys
from datetime import date, timedelta
from io import StringIO
from types import ModuleType
from typing import Annotated, Self, TypeVar

import pytest
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainValidator,
    ValidationError,
    WrapValidator,
    create_model,
    field_validator,
    model_validator,
)

from scheme import StockPool
from scheme.base import (
    ControlAnalysisParams,
    ControlReportForm,
    ExecutionAnalysisParams,
    ExecutionReportForm,
    Factor,
    FactorAnalysisParams,
    FactorParams,
    FactorReportForm,
    ModelAnalysisParams,
    ModelParams,
    ModelReportForm,
    OptimizeAnalysisParams,
    OptimizeReportForm,
    ReportForm,
)
from scheme.manage.parameters import _analysis_type, _form_schema, defaults, inspect_project

CASES = (
    (FactorReportForm, FactorAnalysisParams, {"columns": ["momentum"]}),
    (ModelReportForm, ModelAnalysisParams, {}),
    (OptimizeReportForm, OptimizeAnalysisParams, {"model": "model_example:ModelAlgo"}),
    (ControlReportForm, ControlAnalysisParams, {
        "model": "model_example:ModelAlgo", "optimize": "optimize_example:OptimizeAlgo",
    }),
    (ExecutionReportForm, ExecutionAnalysisParams, {
        "model": "model_example:ModelAlgo", "optimize": "optimize_example:OptimizeAlgo",
        "control": "control_example:ControlAlgo",
    }),
)


@pytest.mark.parametrize("base,analysis,selectors", CASES)
def test_thin_form_automatically_exposes_business_fields(base, analysis, selectors):
    custom = create_model(
        "CustomAnalysis", __base__=analysis,
        window=(int, Field(gt=0, title="窗口", description="交易日", json_schema_extra={"unit": "days"})),
        labels=(list[str], Field(default_factory=lambda: ["base"], min_length=1)),
    )

    class ProjectForm(base[custom]):
        pass

    class InheritedForm(ProjectForm):
        pass

    assert issubclass(ProjectForm, ReportForm)
    assert base[custom] is base[custom]
    assert base[custom] is base[(custom,)]
    assert _analysis_type(ProjectForm, analysis) is custom
    assert _analysis_type(InheritedForm, analysis) is custom
    schema, _ = _form_schema(ProjectForm)
    assert "window" in schema["required"]
    assert schema["properties"]["window"] == {
        "type": "integer", "title": "窗口", "description": "交易日",
        "exclusiveMinimum": 0, "unit": "days",
    }
    assert not {"universe", "config", "symbols"} & schema["properties"].keys()
    initial = defaults(ProjectForm)
    assert initial["labels"] == ["base"]
    assert "window" not in initial
    form = InheritedForm.model_validate({**initial, **selectors, "window": 20})
    built = form.build()
    assert type(built) is custom
    assert built.window == 20
    assert built.start == date(2020, 1, 1)
    assert built.universe.pool == StockPool.CSI300
    form.labels.append("changed")
    assert defaults(ProjectForm)["labels"] == ["base"]
    with pytest.raises(ValidationError, match="window"):
        ProjectForm.model_validate({**selectors, "window": 0}).build()
    with pytest.raises(ValidationError, match="window"):
        ProjectForm.model_validate(selectors)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProjectForm.model_validate({**selectors, "window": 1, "undeclared": 3})


def test_algorithm_decorator_validators_run_during_build():
    calls = []

    class Params(ModelParams):
        window: int = Field(default=20, gt=0)

        @field_validator("window")
        @classmethod
        def check_window(cls, value):
            calls.append("field")
            if value % 2:
                raise ValueError("window 必须为偶数")
            return value

        @model_validator(mode="after")
        def check_settings(self) -> Self:
            calls.append("model")
            if self.config["cash"] < self.window * 100:
                raise ValueError("资金不足")
            return self

    class Analysis(Params, ModelAnalysisParams):
        pass

    form_type = ModelReportForm[Analysis]
    form = form_type(window=3)
    assert calls == []
    with pytest.raises(ValidationError, match="window 必须为偶数"):
        form.build()
    assert calls == ["field"]
    calls.clear()
    with pytest.raises(ValidationError, match="资金不足"):
        form_type(window=20, cash=100).build()
    assert calls == ["field", "model"]
    calls.clear()
    built = form_type(window=20, cash=10_000).build()
    assert type(built) is Analysis
    assert calls == ["field", "model"]


@pytest.mark.parametrize("alias", [
    "days", AliasChoices("days", "legacy_days"),
    AliasChoices(AliasPath("old", "days"), "days"),
])
def test_aliases_and_excluded_fields_survive_build(alias):
    class Analysis(ModelAnalysisParams):
        window: int = Field(default=20, validation_alias=alias, serialization_alias="output_days", gt=0)
        private_input: int = Field(default=7, exclude=True)

    form_type = ModelReportForm[Analysis]
    schema, _ = _form_schema(form_type)
    assert "days" in schema["properties"]
    assert not {"window", "output_days"} & schema["properties"].keys()
    initial = defaults(form_type)
    assert initial["days"] == 20
    built = form_type.model_validate({**initial, "days": 16, "private_input": 9}).build()
    assert type(built) is Analysis
    assert built.window == 16
    assert built.private_input == 9


def test_alias_disabled_model_keeps_canonical_input_keys():
    class Analysis(ModelAnalysisParams):
        model_config = ConfigDict(validate_by_alias=False)
        window: int = Field(default=20, alias="days", gt=0)

    form_type = ModelReportForm[Analysis]
    schema, _ = _form_schema(form_type)
    assert "window" in schema["properties"]
    assert "days" not in schema["properties"]
    assert form_type(window=9).build().window == 9


def test_nested_models_remain_instances_and_keep_subtype_data():
    class Options(BaseModel):
        count: int = Field(alias="input_count", serialization_alias="output_count")

    class DetailedOptions(Options):
        detail: str = "kept"

    class Analysis(ModelAnalysisParams):
        options: Options = Field(default_factory=lambda: DetailedOptions(input_count=4))

    form = ModelReportForm[Analysis]()
    built = form.build()
    assert built.options is not form.options
    assert built.options == form.options
    assert type(built.options) is DetailedOptions
    assert built.options.detail == "kept"
    initial = defaults(type(form))
    assert initial["options"] == {"input_count": 4, "detail": "kept"}


def test_analysis_overrides_replace_fields_without_mutating_shared_defaults():
    class Analysis(FactorAnalysisParams):
        columns: list[str] = Field(default_factory=lambda: ["custom"], min_length=1, title="项目列")
        n_select: int = Field(default=7, ge=3, title="极端样本")
        start: date = Field(default=date(2025, 1, 1), alias="begin")

    form_type = FactorReportForm[Analysis]
    initial = defaults(form_type)
    assert initial["columns"] == ["custom"]
    assert initial["n_select"] == 7
    assert initial["begin"] == "2025-01-01"
    assert initial["weight"] == "market_value"
    assert form_type().build().columns == ["custom"]
    with pytest.raises(ValidationError, match="columns"):
        form_type(columns=[]).build()
    with pytest.raises(ValidationError, match="n_select"):
        form_type(n_select=2).build()
    form_type.model_fields["columns"].json_schema_extra = {"project_only": True}
    assert Analysis.model_fields["columns"].json_schema_extra is None
    assert FactorReportForm.model_fields["columns"].is_required()
    assert defaults(FactorReportForm)["n_select"] == 10


def test_backtest_config_preserves_project_defaults_but_ui_controls_cash_and_fees():
    class Analysis(ModelAnalysisParams):
        config: dict = Field(default_factory=lambda: {"cash": 1, "strategyGroup": "stock"})
        symbols: list[str] | None = ["000001.SZ"]

    built = ModelReportForm[Analysis](cash=50_000, commission=0.001, tax=0.002).build()
    assert built.config == {"cash": 50_000, "commission": 0.001, "tax": 0.002, "strategyGroup": "stock"}
    assert built.symbols == ["000001.XSHE"]


@pytest.mark.parametrize("name", ["pool", "lookback", "cash", "commission", "tax", "build", "analysis_model"])
def test_business_field_names_cannot_shadow_form_conversion_or_methods(name):
    analysis = create_model("CollisionAnalysis", __base__=ModelAnalysisParams, **{name: (int, 3)})
    with pytest.raises(TypeError, match="保留字段冲突"):
        ModelReportForm[analysis]


@pytest.mark.parametrize("alias", [AliasPath("nested", "days"), AliasChoices(AliasPath("nested", 0))])
def test_path_only_alias_fails_explicitly(alias):
    class Analysis(ModelAnalysisParams):
        window: int = Field(default=20, validation_alias=alias)

    with pytest.raises(TypeError, match="AliasPath.*扁平"):
        ModelReportForm[Analysis]


def test_alias_cannot_shadow_another_form_field():
    class Analysis(ModelAnalysisParams):
        window: int = Field(default=20, alias="cash")

    with pytest.raises(TypeError, match="alias.*唯一"):
        ModelReportForm[Analysis]


@pytest.mark.parametrize("target", [FactorAnalysisParams, BaseModel, int, None])
def test_specialization_rejects_wrong_analysis_kind(target):
    with pytest.raises(TypeError, match="ModelAnalysisParams"):
        ModelReportForm[target]


def test_generic_intermediate_binding_and_unbound_rejection():
    assert ModelReportForm[ModelReportForm.__type_params__[0]] is ModelReportForm
    assert ModelReportForm.analysis_model() is ModelAnalysisParams

    class Analysis(ModelAnalysisParams):
        window: int = 20

    class Intermediate[P: ModelAnalysisParams](ModelReportForm[P]):
        pass

    class Concrete(Intermediate[Analysis]):
        pass

    assert _analysis_type(Concrete, ModelAnalysisParams) is Analysis
    assert type(Concrete().build()) is Analysis
    with pytest.raises(TypeError, match="绑定具体"):
        _analysis_type(Intermediate, ModelAnalysisParams)
    variable = TypeVar("Variable", bound=ModelAnalysisParams)
    with pytest.raises(TypeError, match="绑定具体"):
        ModelReportForm[variable]().build()
    with pytest.raises(ValueError, match="FactorAnalysisParams"):
        _analysis_type(Concrete, FactorAnalysisParams)


def test_same_class_name_does_not_share_specialization_or_fields():
    first = create_model("Analysis", __base__=ModelAnalysisParams, window=(int, 10))
    second = create_model("Analysis", __base__=ModelAnalysisParams, count=(int, 3))
    assert ModelReportForm[first] is not ModelReportForm[second]
    assert "window" not in ModelReportForm[second].model_fields
    assert type(ModelReportForm[first]().build()) is first
    assert type(ModelReportForm[second]().build()) is second


def test_optional_ui_overrides_need_no_custom_build():
    class Analysis(ModelAnalysisParams):
        window: int = Field(default=20, ge=1)

    class CustomForm(ModelReportForm[Analysis]):
        cash: float = Field(default=25_000, gt=0)

    assert CustomForm().build().config["cash"] == 25_000
    assert CustomForm().build().window == 20
    assert "build" not in CustomForm.__dict__


@pytest.mark.parametrize("mode", ["after", "before", "plain", "wrap", "list", "alias"])
def test_annotated_conversions_execute_once_in_build_with_original_input_schema(mode):
    calls = []

    def convert(value):
        calls.append(value)
        return int(str(value).removeprefix("raw:")) + 1

    annotation, raw, expected = int, 3, 4
    if mode == "after":
        annotation = Annotated[int, Field(gt=0), AfterValidator(convert)]
    elif mode == "before":
        annotation = Annotated[int, Field(gt=0), BeforeValidator(convert, json_schema_input_type=str)]
        raw = "raw:3"
    elif mode == "plain":
        annotation = Annotated[int, PlainValidator(convert, json_schema_input_type=str)]
        raw = "raw:3"
    elif mode == "wrap":
        annotation = Annotated[int, Field(gt=0), WrapValidator(lambda v, handler: convert(handler(v)))]
    elif mode == "list":
        annotation = list[Annotated[int, Field(gt=0), AfterValidator(convert)]]
        raw, expected = [3], [4]
    else:
        type Window = Annotated[int, Field(gt=0), AfterValidator(convert)]
        annotation = Window
    analysis = create_model("ConvertedAnalysis", __base__=ModelAnalysisParams, window=(annotation, ...))
    form_type = ModelReportForm[analysis]
    def window_schema(model):
        schema = model.model_json_schema()
        field = dict(schema["properties"]["window"])
        if "$ref" in field:
            field = dict(schema["$defs"][field["$ref"].rsplit("/", 1)[-1]])
        field.pop("title", None)
        return field

    assert window_schema(form_type) == window_schema(analysis)
    form = form_type(window=raw)
    assert calls == []
    assert form.build().window == expected
    assert len(calls) == 1
    assert form.window == raw
    assert form.build().window == expected
    assert len(calls) == 2
    if mode != "plain":
        with pytest.raises(ValidationError):
            form_type(window=[-4] if mode == "list" else -4).build()


def test_nested_model_validation_does_not_repeat_or_mutate_the_form():
    class Options(BaseModel):
        count: int

        @model_validator(mode="after")
        def bump(self) -> Self:
            self.count += 1
            return self

    class Analysis(ModelAnalysisParams):
        options: Options

    form = ModelReportForm[Analysis](options={"count": 3})
    assert form.options == {"count": 3}
    assert form.build().options.count == 4
    assert form.build().options.count == 4
    assert form.options == {"count": 3}
    with pytest.raises(ValidationError):
        ModelReportForm[Analysis](options={"count": "bad"}).build()


@pytest.mark.parametrize("alias", [
    AliasChoices("days", "cash"), AliasChoices("days", AliasPath("cash", 0)),
])
def test_secondary_aliases_cannot_capture_platform_values(alias):
    class Analysis(ModelAnalysisParams):
        window: int = Field(default=20, validation_alias=alias)

    with pytest.raises(TypeError, match="alias.*唯一"):
        ModelReportForm[Analysis]


@pytest.mark.parametrize("name", ["window", "config"])
def test_data_dependent_factories_are_rejected_before_schema_discovery(name):
    annotation = dict if name == "config" else int
    field = Field(default_factory=lambda data: {"start": data["start"]} if name == "config" else 20)
    analysis = create_model("DerivedAnalysis", __base__=ModelAnalysisParams, **{name: (annotation, field)})
    with pytest.raises(TypeError, match="default_factory.*model_validator"):
        ModelReportForm[analysis]


@pytest.mark.parametrize("mode", ["before", "plain", "wrap"])
def test_decorator_input_schema_and_raw_input_survive_until_build(mode):
    def decode(value):
        return abs(int(value.removeprefix("raw:")))

    def decode_wrap(value, handler):
        return handler(decode(value))

    validator = field_validator("window", mode=mode, json_schema_input_type=str)(
        decode_wrap if mode == "wrap" else decode
    )
    Analysis = create_model(
        "DecodedAnalysis", __base__=ModelAnalysisParams,
        __validators__={"decode": validator}, window=(int, Field(gt=0)),
    )
    form_type = ModelReportForm[Analysis]
    assert form_type.model_json_schema()["properties"]["window"]["type"] == "string"
    assert form_type(window="raw:-3").build().window == 3


def test_algorithm_strict_and_string_normalization_are_not_bypassed_by_form():
    class Analysis(ModelAnalysisParams):
        model_config = ConfigDict(str_strip_whitespace=True)
        label: str = Field(max_length=3)

    assert ModelReportForm[Analysis](label=" abc ").build().label == "abc"

    class StrictAnalysis(ModelAnalysisParams):
        model_config = ConfigDict(strict=True)
        window: int

    with pytest.raises(ValidationError, match="window"):
        ModelReportForm[StrictAnalysis](window="3").build()
    assert ModelReportForm[StrictAnalysis](window=3).build().window == 3


def test_partial_generic_analysis_must_be_bound_before_generating_form():
    class Analysis[T](ModelAnalysisParams):
        window: T

    for target in (Analysis, Analysis[TypeVar("T")]):
        with pytest.raises(TypeError, match="未绑定泛型"):
            ModelReportForm[target]
    assert ModelReportForm[Analysis[int]](window="3").build().window == 3


def test_real_cli_and_factor_freeze_keep_the_existing_wire_contract(tmp_path, monkeypatch, capsys):
    from scheme.manage import main

    class RuntimeParams(FactorParams):
        window: int = Field(default=20, ge=1, alias="days")

    class Analysis(RuntimeParams, FactorAnalysisParams):
        columns: list[str] = Field(default_factory=lambda: ["momentum"], min_length=1)

    class CustomFactor(Factor[RuntimeParams]):
        def compute(self, start, end):
            raise AssertionError("form tests must not compute")

    class Form(FactorReportForm[Analysis]):
        pass

    module = ModuleType("shared_forms_fixture")
    module.__file__ = str(tmp_path / "src" / module.__name__ / "__init__.py")
    module.Factor, module.FactorReportForm = CustomFactor, Form
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{module.__name__}"\n')
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("scheme.execute.strategy.components.algo_options", lambda _: {})
    assert main(["parameters", "--project", str(tmp_path)]) == 0
    definition = json.loads(capsys.readouterr().out)
    assert definition["protocol"] == 2
    assert definition["values"]["form"]["days"] == 20
    values = {**definition["values"]["form"], "days": 30, "lookback": "P40D"}
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"form": values})))
    assert main(["parameters", "--project", str(tmp_path), "--validate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"entry", "factor", "analysis"}
    assert payload["factor"]["window"] == 30
    assert "window" not in payload["analysis"]
    assert payload["factor"]["universe"] == payload["analysis"]["universe"]
    assert Form.model_validate(values).build().universe.lookback == timedelta(days=40)
    with pytest.raises(ValueError, match="未声明字段"):
        inspect_project(tmp_path, {"form": {**values, "window": 6}})
