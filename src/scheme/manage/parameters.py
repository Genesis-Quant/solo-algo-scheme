"""读取项目的 Pydantic 参数模型；控件由客户端根据 JSON Schema 决定。"""

import importlib
import sys
import tomllib
from importlib import metadata as packages
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import AliasChoices, AliasPath, BaseModel, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import core_schema

from scheme.base import (
    ControlAlgo,
    ControlAnalysisParams,
    ExecutionAlgo,
    ExecutionAnalysisParams,
    Factor,
    FactorAnalysisParams,
    ModelAlgo,
    ModelAnalysisParams,
    OptimizeAlgo,
    OptimizeAnalysisParams,
    ReportForm,
)


def _form_key(model: type[BaseModel], name: str) -> str:
    alias = model.model_fields[name].validation_alias if model.model_config.get("validate_by_alias") is not False else None
    if isinstance(alias, AliasChoices):
        alias = next((choice for choice in alias.choices if isinstance(choice, str)), None)
        if alias is None:
            raise ValueError(f"Form 字段 {name!r} 的 AliasPath 不支持扁平表单 JSON（unsupported）")
    if isinstance(alias, AliasPath):
        raise ValueError(f"Form 字段 {name!r} 的 AliasPath 不支持扁平表单 JSON（unsupported）")
    return alias if isinstance(alias, str) else name


def _form_value(value: Any, annotation: Any = Any) -> Any:
    """Encode defaults with each nested model's validation keys, not serialization aliases."""
    if isinstance(value, BaseModel):
        annotation, value = type(value), dict(value)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType) and value is not None:
        models = [arg for arg in args if isinstance(arg, type) and issubclass(arg, BaseModel)]
        if len(models) == 1 and isinstance(value, dict):
            annotation = models[0]
    if isinstance(value, dict):
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            fields = annotation.model_fields
            wire = {name: _form_key(annotation, name) for name in fields}
            canonical = {key: name for name, key in wire.items()}
            result = {}
            for key, item in value.items():
                name = key if key in fields else canonical.get(key)
                if name is None:
                    result[key] = _form_value(item)  # Unknown inputs must still reach validation.
                else:
                    result[wire[name]] = _form_value(item, fields[name].annotation)
            return result
        item_type = args[1] if origin is dict and len(args) == 2 else Any
        return {key: _form_value(item, item_type) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        item_type = args[0] if args else Any
        return [_form_value(item, item_type) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value  # JSON scalars are already encoded; do not coerce or drop invalid inputs.
    return TypeAdapter(annotation).dump_python(value, mode="json", by_alias=False)


class _FormSchema(GenerateJsonSchema):
    def model_schema(self, schema: core_schema.ModelSchema) -> dict:
        previous = self.by_alias
        self.by_alias = schema["cls"].model_config.get("validate_by_alias") is not False
        try:
            return super().model_schema(schema)
        finally:
            self.by_alias = previous

    def encode_default(self, value: Any) -> Any:
        return super().encode_default(_form_value(value))


def _form_schema(model: type[BaseModel]) -> tuple[dict, dict[str, str]]:
    """Keep Pydantic's refs/required handling, scoped to each model's alias policy."""
    schema = model.model_json_schema(mode="validation", schema_generator=_FormSchema)
    properties = schema.get("properties", {})
    wire = {}
    for name in model.model_fields:
        key = _form_key(model, name)
        if key not in properties or key in wire.values():
            raise ValueError(f"Form 字段 {name!r} 无法唯一对应 schema wire key {key!r}（unsupported）")
        wire[name] = key
    return schema, wire


def defaults(model: type[BaseModel]) -> dict[str, Any]:
    _, wire = _form_schema(model)
    return {
        wire[name]: _form_value(spec.get_default(call_default_factory=True), spec.annotation)
        for name, spec in model.model_fields.items()
        if not spec.is_required()
    }


def _analysis_type(form: type[ReportForm], expected: type[BaseModel]) -> type[BaseModel]:
    """Pydantic 将具体 ReportForm 泛型保存在参数化父类，而非项目子类上。"""
    for base in form.__mro__:
        generic = base.__dict__.get("__pydantic_generic_metadata__", {})
        if generic.get("origin") is ReportForm:
            model = generic["args"][0]
            if isinstance(model, type) and issubclass(model, expected):
                return model
            break
    raise ValueError(f"{form.__name__} 必须声明 ReportForm[{expected.__name__}] 或其分析子类")


def inspect_project(directory: Path, values: dict | None = None) -> dict:
    metadata = tomllib.loads((directory / "pyproject.toml").read_text())
    module_name = metadata.get("tool", {}).get("uv", {}).get("build-backend", {}).get(
        "module-name"
    ) or metadata["project"]["name"].replace("-", "_")
    sys.path.insert(0, str(directory / "src"))
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(directory.resolve()):
        raise ValueError("导入的入口不属于当前源码")
    definitions = (
        (Factor, "FactorReportForm", FactorAnalysisParams),
        (ModelAlgo, "ModelReportForm", ModelAnalysisParams),
        (OptimizeAlgo, "OptimizeReportForm", OptimizeAnalysisParams),
        (ControlAlgo, "ControlReportForm", ControlAnalysisParams),
        (ExecutionAlgo, "ExecutionReportForm", ExecutionAnalysisParams),
    )
    candidates = [
        (algo, form, analysis)
        for algo, form, analysis in definitions
        if hasattr(module, algo.__name__) and hasattr(module, form)
    ]
    if len(candidates) != 1:
        raise ValueError("项目必须导出一组对应类型的算法类和 ReportForm")
    algo_base, form_name, expected_analysis = candidates[0]
    algo_type = getattr(module, algo_base.__name__)
    form_type = getattr(module, form_name)
    if not isinstance(algo_type, type) or not issubclass(algo_type, algo_base):
        raise ValueError(f"{algo_base.__name__} 必须继承 Scheme 对应基类")
    if not isinstance(form_type, type) or not issubclass(form_type, ReportForm):
        raise ValueError(f"{form_name} 必须继承 ReportForm")
    analysis_type = _analysis_type(form_type, expected_analysis)
    schema, wire = _form_schema(form_type)
    if values is not None:
        unknown = set(values["form"]) - set(wire.values())
        if unknown:
            raise ValueError(f"{form_name} 包含未声明字段：{sorted(unknown)}")
        analysis = form_type.model_validate(values["form"]).build()
        if not isinstance(analysis, analysis_type):
            raise ValueError(f"{form_name}.build() 必须返回 {analysis_type.__name__}")
        if algo_base is not Factor:
            from scheme.execute.strategy.components import BASES, research_components

            kind = next(name for name, base in BASES.items() if base is algo_base)
            components = research_components(analysis, kind, algo_type)
            upstream = {}
            for name in components:
                if name == kind:
                    continue
                entry = getattr(analysis, name)
                package = entry.split(":")[0].replace("_", "-")
                dist = packages.distribution(package)
                upstream[name] = {
                    "package": dist.metadata["Name"],
                    "version": dist.version,
                    "entry": entry,
                }
            return {
                "entry": f"{module_name}:{algo_base.__name__}",
                "project_kind": kind,
                "upstream": upstream,
                "backtest": analysis.model_dump(mode="json"),
            }
        from scheme.execute.strategy.assembly import parameter_type, project_parameters

        factor_model = parameter_type(algo_type)
        unsupported = (
            set(type(analysis).model_fields) | set(analysis.model_extra or {})
        ) - factor_model.model_fields.keys() - FactorAnalysisParams.model_fields.keys()
        if unsupported:
            raise ValueError(
                f"因子分析自定义字段无法由标准分析器或 Factor 参数保存：{sorted(unsupported)}"
            )
        return {
            "entry": f"{module_name}:Factor",
            "factor": project_parameters(factor_model, analysis).model_dump(mode="json"),
            "analysis": project_parameters(FactorAnalysisParams, analysis).model_dump(mode="json"),
        }

    from scheme.execute.strategy.components import algo_options

    for field in schema.get("properties", {}).values():
        if kind := field.get("x-algo-kind"):
            options = algo_options(kind)
            field.update(enum=list(options), **{"x-enum-labels": list(options.values())})
    return {
        "protocol": 2,
        "title": "保存版本 · 研究参数",
        "schemas": {"form": schema},
        "values": {"form": defaults(form_type)},
    }
