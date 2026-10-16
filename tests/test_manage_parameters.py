"""项目表单与冻结参数协议；因子保存校验使用离线替身，不运行研究任务。"""

import json
import sys
from datetime import date
from io import StringIO
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, ValidationError

from scheme.base import (
    ControlAlgo,
    ControlAnalysisParams,
    ExecutionAlgo,
    ExecutionAnalysisParams,
    Factor,
    FactorAnalysisParams,
    FactorParams,
    ModelAlgo,
    ModelAnalysisParams,
    OptimizeAlgo,
    OptimizeAnalysisParams,
    ReportForm,
)
from scheme.manage.parameters import defaults, inspect_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    module = ModuleType("manage_parameters_fixture")
    module.__file__ = str(tmp_path / "src" / module.__name__ / "__init__.py")
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{module.__name__}"\n')
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("scheme.execute.strategy.components.algo_options", lambda _: {})
    return tmp_path, module


class RuntimeParams(FactorParams):
    window: int = Field(gt=0)


class CustomFactor(Factor[RuntimeParams]):
    def compute(self, start, end):
        raise AssertionError("parameter inspection must not compute")


class CustomAnalysis(FactorAnalysisParams, RuntimeParams):
    def run(self, *args, **kwargs):
        raise AssertionError("parameter inspection must not run analysis")


class CustomForm(ReportForm[CustomAnalysis]):
    start: date
    end: date
    window: int

    def build(self) -> CustomAnalysis:
        return CustomAnalysis(**self.model_dump(), columns=["f"])


@pytest.mark.parametrize("end", ["2026-06-01", "2026-06-03"])
def test_direct_generic_form_preserves_custom_factor_runtime_fields(project, monkeypatch, end):
    directory, module = project
    module.Factor, module.FactorReportForm = CustomFactor, CustomForm
    construct = Mock(side_effect=AssertionError("schema inspection must not instantiate a factor"))
    monkeypatch.setattr(CustomFactor, "__init__", construct)
    definition = inspect_project(directory)
    assert definition["protocol"] == 2
    assert set(definition["schemas"]) == {"form"}
    assert set(definition["schemas"]["form"]["required"]) == {"start", "end", "window"}
    assert definition["values"] == {"form": {}}
    payload = inspect_project(directory, {"form": {
        "start": "2026-06-01", "end": end, "window": 17,
    }})
    assert set(payload) == {"entry", "factor", "analysis"}
    assert payload["entry"] == f"{module.__name__}:Factor"
    assert set(payload["factor"]) == {"universe", "window"}
    assert RuntimeParams.model_validate(payload["factor"]).window == 17
    canonical = FactorAnalysisParams.model_validate(payload["analysis"])
    assert canonical.columns == ["f"]
    assert (canonical.start, canonical.end) == (date(2026, 6, 1), date.fromisoformat(end))
    assert "window" not in payload["analysis"]
    assert payload["factor"]["universe"] == payload["analysis"]["universe"]
    construct.assert_not_called()


@pytest.mark.parametrize("end", ["2026-06-01", "2026-06-03"])
def test_save_factor_validation_instantiates_concrete_runtime_once(project, end):
    directory, module = project
    calls = []

    class ValidatingFactor(CustomFactor):
        def __init__(self, params):
            assert type(params) is RuntimeParams
            assert set(params.model_dump()) == {"universe", "window"}
            calls.append(("init", params))
            super().__init__(params)

        def validate(self, start, end, n_sample=5):
            calls.append(("validate", start, end, n_sample))

    module.Factor, module.FactorReportForm = ValidatingFactor, CustomForm
    payload = inspect_project(directory, {"form": {
        "start": "2026-06-01", "end": end, "window": 17,
    }}, validate_factor=True)
    assert len(calls) == 2
    assert calls[0][0] == "init"
    assert calls[0][1].window == 17
    assert calls[1] == ("validate", date(2026, 6, 1), date.fromisoformat(end), 5)
    assert payload["factor"] == calls[0][1].model_dump(mode="json")
    assert payload["analysis"]["start"] == "2026-06-01"
    assert payload["analysis"]["end"] == end


@pytest.mark.parametrize("form", [
    {},
    {"start": "2026-06-01", "end": "2026-06-03", "window": 0},
    {"start": "2026-06-03", "end": "2026-06-01", "window": 17},
])
def test_save_factor_validation_rejects_invalid_form_before_instantiation(project, monkeypatch, form):
    directory, module = project
    module.Factor, module.FactorReportForm = CustomFactor, CustomForm
    construct = Mock(side_effect=AssertionError("invalid form instantiated a factor"))
    monkeypatch.setattr(CustomFactor, "__init__", construct)
    with pytest.raises(ValidationError):
        inspect_project(directory, {"form": form}, validate_factor=True)
    construct.assert_not_called()


def test_save_factor_validation_rejects_runtime_projection_before_instantiation(project, monkeypatch):
    from pydantic import field_validator

    directory, module = project

    class EvenRuntimeParams(RuntimeParams):
        @field_validator("window")
        @classmethod
        def even_window(cls, value):
            if value % 2:
                raise ValueError("runtime window must be even")
            return value

    class EvenFactor(Factor[EvenRuntimeParams]):
        def compute(self, start, end):
            raise AssertionError("invalid runtime params must not compute")

    module.Factor, module.FactorReportForm = EvenFactor, CustomForm
    construct = Mock(side_effect=AssertionError("invalid runtime params instantiated a factor"))
    monkeypatch.setattr(EvenFactor, "__init__", construct)
    with pytest.raises(ValidationError, match="runtime window must be even"):
        inspect_project(directory, {"form": {
            "start": "2026-06-01", "end": "2026-06-03", "window": 17,
        }}, validate_factor=True)
    construct.assert_not_called()


@pytest.mark.parametrize("alias", [
    "lookback_days",
    AliasChoices("lookback_days", "legacy_days"),
    AliasChoices(AliasPath("config", "window"), "lookback_days", "legacy_days"),
])
def test_sole_generic_form_alias_defaults_round_trip_to_runtime(project, alias):
    directory, module = project

    class AliasedForm(ReportForm[CustomAnalysis]):
        model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)
        start: date = date(2026, 6, 1)
        end: date = date(2026, 6, 3)
        window: int = Field(
            default=20, validation_alias=alias, serialization_alias="serialized_days", gt=0,
        )

        def build(self) -> CustomAnalysis:
            return CustomAnalysis(
                start=self.start, end=self.end, window=self.window, columns=["f"],
            )

    module.Factor, module.FactorReportForm = CustomFactor, AliasedForm
    definition = inspect_project(directory)
    assert definition["protocol"] == 2
    assert set(definition["schemas"]) == {"form"}
    schema = definition["schemas"]["form"]
    assert set(schema["properties"]) == {"start", "end", "lookback_days"}
    assert schema["properties"]["lookback_days"]["default"] == 20
    assert schema.get("required", []) == []
    assert definition["values"] == {"form": {
        "start": "2026-06-01", "end": "2026-06-03", "lookback_days": 20,
    }}
    payload = inspect_project(directory, {
        "form": {**definition["values"]["form"], "lookback_days": 17},
    })
    assert payload["factor"]["window"] == 17
    assert "window" not in payload["analysis"]
    for key in ("window", "serialized_days", "legacy_days", "config", "unknown", "model"):
        assert inspect_project(directory, {
            "form": {"lookback_days": 17, key: 99},
        })["factor"]["window"] == 17
    assert inspect_project(directory, {"form": {"window": 17}})["factor"]["window"] == 17
    if isinstance(alias, AliasChoices):
        assert inspect_project(directory, {"form": {"legacy_days": 17}})["factor"]["window"] == 17
        if isinstance(alias.choices[0], AliasPath):
            assert inspect_project(directory, {
                "form": {"config": {"window": 17}},
            })["factor"]["window"] == 17
    with pytest.raises(ValidationError, match="lookback_days"):
        inspect_project(directory, {"form": {"lookback_days": 0}})


def test_required_form_alias_is_published_and_validated(project):
    directory, module = project

    class AliasedForm(ReportForm[CustomAnalysis]):
        window: int = Field(alias="lookback_days")

        def build(self) -> CustomAnalysis:
            return CustomAnalysis(
                start="2026-06-01", end="2026-06-03", window=self.window, columns=["f"],
            )

    module.Factor, module.FactorReportForm = CustomFactor, AliasedForm
    definition = inspect_project(directory)
    assert definition["schemas"]["form"]["required"] == ["lookback_days"]
    assert definition["values"] == {"form": {}}
    assert inspect_project(directory, {"form": {"lookback_days": 17}})["factor"]["window"] == 17
    with pytest.raises(ValidationError, match="lookback_days"):
        inspect_project(directory, {"form": {}})


@pytest.mark.parametrize("alias", [
    AliasPath("config", "window"),
    AliasPath("lookback_days"),
    AliasChoices(AliasPath("config", "window"), AliasPath("other", 0)),
])
@pytest.mark.parametrize("submit", [False, True])
def test_form_alias_path_without_flat_schema_key_is_unsupported(project, alias, submit):
    directory, module = project

    class UnsupportedForm(ReportForm[CustomAnalysis]):
        window: int = Field(default=20, validation_alias=alias)

        def build(self) -> CustomAnalysis:
            raise AssertionError("unsupported form must not build")

    module.Factor, module.FactorReportForm = CustomFactor, UnsupportedForm
    with pytest.raises(ValueError, match="window.*AliasPath.*unsupported"):
        inspect_project(directory, {"form": {"window": 17}} if submit else None)


def test_form_with_alias_validation_disabled_keeps_canonical_wire_key(project):
    directory, module = project

    class CanonicalForm(ReportForm[CustomAnalysis]):
        model_config = ConfigDict(validate_by_alias=False)
        window: int = Field(default=20, alias="lookback_days")

        def build(self) -> CustomAnalysis:
            return CustomAnalysis(
                start="2026-06-01", end="2026-06-03", window=self.window, columns=["f"],
            )

    module.Factor, module.FactorReportForm = CustomFactor, CanonicalForm
    definition = inspect_project(directory)
    assert set(definition["schemas"]["form"]["properties"]) == {"window"}
    assert definition["values"] == {"form": {"window": 20}}
    assert inspect_project(directory, {"form": {"window": 17}})["factor"]["window"] == 17
    with pytest.raises(ValidationError, match="lookback_days.*[\\s\\S]*extra_forbidden"):
        inspect_project(directory, {"form": {"lookback_days": 17}})


def test_defaults_preserve_default_factories_and_json_values():
    class Form(BaseModel):
        tags: list[str] = Field(default_factory=list, alias="labels")
        start: date = date(2026, 6, 1)

    first, second = defaults(Form), defaults(Form)
    assert first == second == {"labels": [], "start": "2026-06-01"}
    assert first["labels"] is not second["labels"]


def test_manage_cli_keeps_single_generic_form_protocol(project, monkeypatch, capsys):
    from scheme.manage import main

    directory, module = project

    class CliForm(ReportForm[CustomAnalysis]):
        window: int = Field(default=20, alias="lookback_days")

        def build(self) -> CustomAnalysis:
            return CustomAnalysis(
                start="2026-06-01", end="2026-06-03", window=self.window, columns=["f"],
            )

    module.Factor, module.FactorReportForm = CustomFactor, CliForm
    assert main(["parameters", "--project", str(directory)]) == 0
    definition = json.loads(capsys.readouterr().out)
    assert definition["protocol"] == 2
    assert set(definition["schemas"]) == {"form"}
    assert definition["values"] == {"form": {"lookback_days": 20}}
    monkeypatch.setattr(sys, "stdin", StringIO('{"form":{"lookback_days":17}}'))
    assert main(["parameters", "--project", str(directory), "--validate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entry"] == f"{module.__name__}:Factor"
    assert payload["factor"]["window"] == 17


def test_manage_cli_save_factor_validation_uses_analysis_dates_once(project, monkeypatch, capsys):
    from scheme.manage import main

    directory, module = project
    calls = []

    class ValidatingFactor(CustomFactor):
        def validate(self, start, end, n_sample=5):
            print("user factor validation output")
            calls.append((self.params, start, end, n_sample))

    module.Factor, module.FactorReportForm = ValidatingFactor, CustomForm
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"form": {
        "start": "2026-06-01", "end": "2026-06-01", "window": 17,
    }})))
    assert main([
        "parameters", "--project", str(directory), "--validate", "--validate-factor",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines()[0] == "user factor validation output"
    payload = json.loads(captured.out.splitlines()[-1])
    assert len(calls) == 1
    params, start, end, n_sample = calls[0]
    assert type(params) is RuntimeParams
    assert (start, end, n_sample) == (date(2026, 6, 1), date(2026, 6, 1), 5)
    assert payload["factor"] == params.model_dump(mode="json")
    assert payload["analysis"]["start"] == payload["analysis"]["end"] == "2026-06-01"


def test_manage_cli_factor_validation_failure_is_nonzero_without_success_payload(
    project, monkeypatch, capsys,
):
    from scheme.manage import main

    directory, module = project
    calls = []

    class InconsistentFactor(CustomFactor):
        def validate(self, start, end, n_sample=5):
            calls.append((start, end, n_sample))
            print("user factor validation output")
            raise ValueError("factor differs on sampled date")

    module.Factor, module.FactorReportForm = InconsistentFactor, CustomForm
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"form": {
        "start": "2026-06-01", "end": "2026-06-03", "window": 17,
    }})))
    assert main([
        "parameters", "--project", str(directory), "--validate", "--validate-factor",
    ]) == 1
    captured = capsys.readouterr()
    assert calls == [(date(2026, 6, 1), date(2026, 6, 3), 5)]
    assert captured.err == "ValueError: factor differs on sampled date\n"
    assert captured.out == "user factor validation output\n"
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out.splitlines()[-1])


@pytest.mark.parametrize("stdin", ["null", "[]", "0", "{invalid json"])
def test_manage_cli_factor_validation_rejects_non_form_json_without_success_payload(
    project, monkeypatch, capsys, stdin,
):
    from scheme.manage import main

    directory, _ = project
    inspect = Mock(side_effect=AssertionError("invalid JSON inspected a project"))
    monkeypatch.setattr("scheme.manage.parameters.inspect_project", inspect)
    monkeypatch.setattr(sys, "stdin", StringIO(stdin))
    assert main([
        "parameters", "--project", str(directory), "--validate", "--validate-factor",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
    inspect.assert_not_called()


def test_manage_cli_factor_validation_requires_method_without_legacy_fallback(
    project, monkeypatch, capsys,
):
    from scheme.manage import main

    directory, module = project
    module.Factor, module.FactorReportForm = CustomFactor, CustomForm
    monkeypatch.delattr(Factor, "validate")
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"form": {
        "start": "2026-06-01", "end": "2026-06-03", "window": 17,
    }})))
    assert main([
        "parameters", "--project", str(directory), "--validate", "--validate-factor",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "AttributeError" in captured.err
    assert "validate" in captured.err


@pytest.mark.parametrize("arguments", [
    ["parameters", "--validate-factor"],
    ["run", "--validate-factor"],
    ["run", "--validate", "--validate-factor"],
])
def test_manage_cli_factor_validation_requires_parameters_json_validation(
    monkeypatch, capsys, arguments,
):
    from scheme.manage import main

    inspect = Mock(side_effect=AssertionError("invalid flag combination inspected a project"))
    monkeypatch.setattr("scheme.manage.parameters.inspect_project", inspect)
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--validate-factor 仅可与 parameters --validate" in captured.err
    inspect.assert_not_called()


@pytest.mark.parametrize("updates", [{}, {"window": 0}])
def test_required_custom_factor_field_is_validated(project, updates):
    directory, module = project
    module.Factor, module.FactorReportForm = CustomFactor, CustomForm
    with pytest.raises(ValidationError, match="window"):
        inspect_project(directory, {"form": {
            "start": "2026-06-01", "end": "2026-06-03", **updates,
        }})


@pytest.mark.parametrize("wrong_type", [dict, FactorAnalysisParams, ModelAnalysisParams])
def test_build_must_return_the_declared_matching_analysis(project, wrong_type):
    directory, module = project

    class WrongForm(ReportForm[CustomAnalysis]):
        def build(self) -> CustomAnalysis:
            return wrong_type(start="2026-06-01", end="2026-06-03", columns=["f"])

    module.Factor, module.FactorReportForm = CustomFactor, WrongForm
    with pytest.raises(ValueError, match=r"build\(\).*CustomAnalysis"):
        inspect_project(directory, {"form": {}})


def test_form_generic_must_match_project_kind(project):
    directory, module = project

    class WrongForm(ReportForm[ModelAnalysisParams]):
        def build(self) -> ModelAnalysisParams:
            raise AssertionError("schema inspection must not build")

    module.Factor, module.FactorReportForm = CustomFactor, WrongForm
    with pytest.raises(ValueError, match=r"ReportForm\[FactorAnalysisParams\]"):
        inspect_project(directory)


@pytest.mark.parametrize("base", [BaseModel, ReportForm])
def test_form_must_declare_report_form_analysis_generic(project, base):
    directory, module = project

    class UnspecifiedForm(base):
        def build(self):
            raise AssertionError("schema inspection must not build")

    module.Factor, module.FactorReportForm = CustomFactor, UnspecifiedForm
    with pytest.raises(ValueError, match="ReportForm"):
        inspect_project(directory)


@pytest.mark.parametrize("validate_factor", [False, True])
def test_unrepresentable_factor_analysis_fields_fail_clearly(project, monkeypatch, validate_factor):
    directory, module = project
    construct = Mock(side_effect=AssertionError("unsupported analysis instantiated a factor"))
    monkeypatch.setattr(CustomFactor, "__init__", construct)

    class UnsupportedAnalysis(CustomAnalysis):
        evaluator: str

    class UnsupportedForm(ReportForm[UnsupportedAnalysis]):
        def build(self) -> UnsupportedAnalysis:
            return UnsupportedAnalysis(
                start="2026-06-01", end="2026-06-03", columns=["f"],
                window=17, evaluator="custom",
            )

    module.Factor, module.FactorReportForm = CustomFactor, UnsupportedForm
    with pytest.raises(ValueError, match="自定义字段.*evaluator"):
        inspect_project(directory, {"form": {}}, validate_factor=validate_factor)
    construct.assert_not_called()


@pytest.mark.parametrize("validate_factor", [False, True])
@pytest.mark.parametrize("kind,algo,analysis_type", [
    ("model", ModelAlgo, ModelAnalysisParams),
    ("optimize", OptimizeAlgo, OptimizeAnalysisParams),
    ("control", ControlAlgo, ControlAnalysisParams),
    ("execution", ExecutionAlgo, ExecutionAnalysisParams),
])
def test_direct_stage_form_keeps_backtest_wire_shape(
    project, monkeypatch, kind, algo, analysis_type, validate_factor,
):
    directory, module = project
    construct = Mock(side_effect=AssertionError("stage project instantiated a factor"))
    monkeypatch.setattr(Factor, "__init__", construct)
    stages = ("model", "optimize", "control", "execution")
    selectors = {name: f"{name}_example:{name.title()}Algo"
                 for name in stages[:stages.index(kind)]}

    class StageForm(ReportForm[analysis_type]):
        start: date
        end: date

        def build(self) -> analysis_type:
            return analysis_type(start=self.start, end=self.end, **selectors)

    calls = []

    def components(analysis, selected_kind, selected_algo):
        calls.append((analysis, selected_kind, selected_algo))
        return {kind: algo}

    setattr(module, algo.__name__, algo)
    setattr(module, f"{kind.title()}ReportForm", StageForm)
    monkeypatch.setattr("scheme.execute.strategy.components.research_components", components)
    payload = inspect_project(directory, {"form": {
        "start": "2026-06-01", "end": "2026-06-03",
    }}, validate_factor=validate_factor)
    construct.assert_not_called()
    assert set(payload) == {"entry", "project_kind", "upstream", "backtest"}
    assert payload["entry"] == f"{module.__name__}:{algo.__name__}"
    assert payload["project_kind"] == kind
    assert payload["upstream"] == {}
    assert payload["backtest"] == calls[0][0].model_dump(mode="json")
    assert calls[0][1:] == (kind, algo)
    assert type(calls[0][0]) is analysis_type


def test_upstream_schema_metadata_and_package_wire_shape(project, monkeypatch):
    directory, module = project

    class Form(ReportForm[OptimizeAnalysisParams]):
        model: str = Field(json_schema_extra={"x-algo-kind": "model"})

        def build(self) -> OptimizeAnalysisParams:
            return OptimizeAnalysisParams(
                start="2026-06-01", end="2026-06-03", model=self.model,
            )

    module.OptimizeAlgo, module.OptimizeReportForm = OptimizeAlgo, Form
    monkeypatch.setattr(
        "scheme.execute.strategy.components.algo_options",
        lambda _: {"model_example:ModelAlgo": "Example · 1.0.0"},
    )
    field = inspect_project(directory)["schemas"]["form"]["properties"]["model"]
    assert field["x-algo-kind"] == "model"
    assert field["enum"] == ["model_example:ModelAlgo"]
    assert field["x-enum-labels"] == ["Example · 1.0.0"]
    monkeypatch.setattr(
        "scheme.execute.strategy.components.research_components",
        lambda *args: {"model": ModelAlgo, "optimize": OptimizeAlgo},
    )
    monkeypatch.setattr(
        "scheme.manage.parameters.packages.distribution",
        lambda name: SimpleNamespace(metadata={"Name": name}, version="1.0.0"),
    )
    payload = inspect_project(directory, {"form": {"model": "model_example:ModelAlgo"}})
    assert payload["upstream"] == {"model": {
        "package": "model-example", "version": "1.0.0", "entry": "model_example:ModelAlgo",
    }}


@pytest.mark.parametrize("outer_alias", [False, True])
@pytest.mark.parametrize("nested_alias", [False, True])
def test_nested_form_alias_scopes_round_trip_schema_defaults_and_submission(
    project, outer_alias, nested_alias,
):
    directory, module = project

    class Nested(BaseModel):
        model_config = ConfigDict(validate_by_alias=nested_alias, extra="forbid")
        count: int = Field(
            validation_alias=AliasChoices("n_select", "legacy_count"),
            serialization_alias="reported_count", gt=0,
        )

    nested_key = "n_select" if nested_alias else "count"

    class Form(ReportForm[CustomAnalysis]):
        model_config = ConfigDict(validate_by_alias=outer_alias)
        nested: Nested = Field(alias="selection")
        optional: Nested | None = Field(
            default=Nested(**{nested_key: 7}), alias="optional_selection",
        )
        items: list[Nested] = Field(
            default_factory=lambda: [Nested(**{nested_key: 11})], alias="selections",
        )
        empty: Nested | None = Field(default=None, alias="empty_selection")

        def build(self) -> CustomAnalysis:
            return CustomAnalysis(
                start="2026-06-01", end="2026-06-03", columns=["f"],
                window=self.nested.count + self.optional.count + sum(item.count for item in self.items),
            )

    module.Factor, module.FactorReportForm = CustomFactor, Form
    definition = inspect_project(directory)
    assert definition["protocol"] == 2
    schema = definition["schemas"]["form"]
    nested_schema = schema["$defs"]["Nested"]
    assert set(nested_schema["properties"]) == {nested_key}
    assert nested_schema["required"] == [nested_key]
    keys = {
        "nested": "selection", "optional": "optional_selection",
        "items": "selections", "empty": "empty_selection",
    } if outer_alias else {name: name for name in Form.model_fields}
    assert set(schema["properties"]) == set(keys.values())
    assert schema["required"] == [keys["nested"]]
    assert schema["properties"][keys["optional"]]["default"] == {nested_key: 7}
    assert schema["properties"][keys["nested"]]["$ref"] == "#/$defs/Nested"
    assert schema["properties"][keys["items"]]["items"]["$ref"] == "#/$defs/Nested"
    assert definition["values"] == {"form": {
        keys["optional"]: {nested_key: 7}, keys["items"]: [{nested_key: 11}],
        keys["empty"]: None,
    }}
    values = {**definition["values"]["form"], keys["nested"]: {nested_key: 17}}
    assert inspect_project(directory, {"form": values})["factor"]["window"] == 35
    with pytest.raises(ValidationError, match=nested_key):
        inspect_project(directory, {"form": {**values, keys["nested"]: {}}})
    with pytest.raises(ValidationError):
        inspect_project(directory, {"form": {
            **values, keys["nested"]: {"count" if nested_alias else "n_select": 17},
        }})
    with pytest.raises(ValidationError, match="undeclared"):
        inspect_project(directory, {"form": {
            **values, keys["nested"]: {nested_key: 17, "undeclared": 99},
        }})


def test_nested_mixed_config_recurses_without_changing_ordinary_form_schema():
    class Aliased(BaseModel):
        count: int = Field(alias="n_select")

    class Canonical(BaseModel):
        model_config = ConfigDict(validate_by_alias=False, extra="forbid")
        nested: Aliased = Field(alias="hidden_nested")
        count: int = Field(alias="hidden_count")

    class Form(BaseModel):
        settings: Canonical = Canonical(nested=Aliased(n_select=7), count=11)

    from scheme.manage.parameters import _form_schema

    schema, wire = _form_schema(Form)
    assert wire == {"settings": "settings"}
    assert set(schema["$defs"]["Canonical"]["properties"]) == {"nested", "count"}
    assert set(schema["$defs"]["Aliased"]["properties"]) == {"n_select"}
    values = {"settings": {"nested": {"n_select": 7}, "count": 11}}
    assert defaults(Form) == values
    assert schema["properties"]["settings"]["default"] == values["settings"]
    assert Form.model_validate(values).settings.count == 11

    for ordinary in (CustomForm,):
        assert _form_schema(ordinary)[0] == ordinary.model_json_schema(mode="validation")
