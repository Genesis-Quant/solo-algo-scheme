"""读取项目的 Pydantic 参数模型；控件由客户端根据 JSON Schema 决定。"""

import importlib
import sys
import tomllib
from importlib import metadata as packages
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

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


def defaults(model: type[BaseModel]) -> dict[str, Any]:
    return {
        name: TypeAdapter(spec.annotation).dump_python(
            spec.get_default(call_default_factory=True), mode="json"
        )
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
    if values is not None:
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

    schema = form_type.model_json_schema()
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
