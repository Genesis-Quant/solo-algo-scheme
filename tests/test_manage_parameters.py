"""项目表单发现与冻结参数序列化；不运行算法或研究任务。"""

import json
import sys
from datetime import date
from io import StringIO
from types import ModuleType, SimpleNamespace

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


def test_direct_generic_form_preserves_custom_factor_runtime_fields(project):
    directory, module = project
    module.Factor, module.FactorReportForm = CustomFactor, CustomForm
    definition = inspect_project(directory)
    assert definition["protocol"] == 2
    assert set(definition["schemas"]) == {"form"}
    assert set(definition["schemas"]["form"]["required"]) == {"start", "end", "window"}
    assert definition["values"] == {"form": {}}
    payload = inspect_project(directory, {"form": {
        "start": "2026-06-01", "end": "2026-06-03", "window": 17,
    }})
    assert set(payload) == {"entry", "factor", "analysis"}
    assert payload["entry"] == f"{module.__name__}:Factor"
    assert set(payload["factor"]) == {"start", "end", "universe", "window"}
    assert RuntimeParams.model_validate(payload["factor"]).window == 17
    canonical = FactorAnalysisParams.model_validate(payload["analysis"])
    assert canonical.columns == ["f"]
    assert "window" not in payload["analysis"]
    for name in ("start", "end", "universe"):
        assert payload["factor"][name] == payload["analysis"][name]


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
    for unknown in ("window", "serialized_days", "legacy_days", "config", "unknown", "model"):
        with pytest.raises(ValueError, match=f"未声明字段.*{unknown}"):
            inspect_project(directory, {"form": {"lookback_days": 17, unknown: 99}})
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
    with pytest.raises(ValueError, match="未声明字段.*lookback_days"):
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


def test_unrepresentable_factor_analysis_fields_fail_clearly(project):
    directory, module = project

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
        inspect_project(directory, {"form": {}})


@pytest.mark.parametrize("kind,algo,analysis_type", [
    ("model", ModelAlgo, ModelAnalysisParams),
    ("optimize", OptimizeAlgo, OptimizeAnalysisParams),
    ("control", ControlAlgo, ControlAnalysisParams),
    ("execution", ExecutionAlgo, ExecutionAnalysisParams),
])
def test_direct_stage_form_keeps_backtest_wire_shape(
    project, monkeypatch, kind, algo, analysis_type,
):
    directory, module = project
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
    }})
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
