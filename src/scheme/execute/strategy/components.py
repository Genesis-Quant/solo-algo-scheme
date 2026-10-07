"""从当前环境的研究包选择算法；正式任务另行校验冻结 wheel。"""

import importlib
from importlib import metadata
from typing import TYPE_CHECKING, Any, cast

from packaging.version import Version
from pydantic import BaseModel

from scheme.base import (
    Algo,
    ControlAlgo,
    ExecutionAlgo,
    ModelAlgo,
    OptimizeAlgo,
    ResearchContext,
    StrategyAnalysisParams,
)

if TYPE_CHECKING:
    from scheme.config import DolphinSettings

    from .result import BacktestResult

BASES = {
    "model": ModelAlgo,
    "optimize": OptimizeAlgo,
    "control": ControlAlgo,
    "execution": ExecutionAlgo,
}

__all__ = ["algo_options"]


def algo_options(kind: str) -> dict[str, str]:
    """返回已安装同 Scheme major.minor 项目的 {类入口: 包名及版本} 选项。"""
    from scheme.execute.packages import validate_scheme_requirement

    if kind not in BASES:
        raise ValueError(f"未知 Algo 类型：{kind}")
    version = Version(metadata.version("scheme"))
    options = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        module = name.replace("-", "_")
        # 平台生成的包名为 <类型>-<项目 ID>，模板按约定导出同名 Algo。
        if module != kind and not module.startswith(kind + "_"):
            continue
        try:
            validate_scheme_requirement(dist.requires or [], version, Version(dist.version))
        except ValueError:
            continue
        entry = f"{module}:{BASES[kind].__name__}"
        options[entry] = f"{name} · {dist.version}"
    return dict(sorted(options.items()))


def resolve_algo(kind: str, entry: str) -> type:
    if entry not in algo_options(kind):
        raise ValueError(f"{kind} 上游未安装或 Scheme 版本不兼容：{entry}")
    module, name = entry.split(":")
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, BASES[kind]):
        raise TypeError(f"{entry} 必须继承 {BASES[kind].__name__}")
    return cls


def research_components(
    params: BaseModel,
    kind: str,
    current: type[Algo[Any, Any]],
) -> dict[str, type[Algo[Any, Any]]]:
    """解析已安装的上游和当前环节；后续算法由分析参数明确选择。"""
    base = BASES[kind]
    if not isinstance(current, type) or not issubclass(current, base):
        raise TypeError(f"run 需要 {base.__name__} 类")
    components = {}
    for upstream in list(BASES)[: list(BASES).index(kind)]:
        components[upstream] = resolve_algo(upstream, getattr(params, upstream))
    components[kind] = current
    return components


def run_research[A: BaseModel, P: BaseModel, C: ResearchContext[Any]](
    params: A,
    kind: str,
    current: type[Algo[P, C]],
    *,
    ctx: C | None = None,
    settings: "DolphinSettings | None" = None,
) -> "BacktestResult[A]":
    from scheme.base.projects.control.default import NoControl
    from scheme.base.projects.execution.default import DirectExecution
    from scheme.base.projects.optimize.default import RiskParity

    from .api import run_backtest
    from .assembly import context_type, parameter_type, project_parameters, validate_context

    components = research_components(params, kind, current)
    defaults = {
        "optimize": {"risk_parity": RiskParity},
        "control": {"no_control": NoControl},
        "execution": {"direct_execution": DirectExecution},
    }
    for downstream in list(BASES)[list(BASES).index(kind) + 1:]:
        components[downstream] = defaults[downstream][getattr(params, downstream)]
    context = ctx if ctx is not None else context_type(components["model"])()
    backtest = StrategyAnalysisParams.model_validate(dict(params))
    algos = tuple(cls(project_parameters(parameter_type(cls), params)) for cls in components.values())
    for algo in algos:
        validate_context(algo, context)
    result = cast("BacktestResult[A]", run_backtest(algos, context, backtest, settings=settings))
    result.parameters = params.model_copy(deep=True)
    return result
