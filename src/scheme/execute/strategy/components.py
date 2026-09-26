"""从当前环境的研究包选择算法；正式任务另行校验冻结 wheel。"""

import importlib
from importlib import metadata
from typing import TYPE_CHECKING, Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from scheme.base import (
    Algo,
    BacktestParameters,
    ControlAlgo,
    ExecutionAlgo,
    ModelAlgo,
    OptimizeAlgo,
    ResearchContext,
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
    """返回已安装同 Scheme 大版本项目的 {类入口: 包名及版本} 选项。"""
    if kind not in BASES:
        raise ValueError(f"未知 Algo 类型：{kind}")
    major = Version(metadata.version("scheme")).major
    options = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        module = name.replace("-", "_")
        # 平台生成的包名为 <类型>-<项目 ID>，模板按约定导出同名 Algo。
        if (module != kind and not module.startswith(kind + "_")) or (
            Version(dist.version).major != major
        ):
            continue
        requirements = [Requirement(value) for value in dist.requires or []]
        if not any(
            canonicalize_name(value.name) == "scheme"
            and metadata.version("scheme") in value.specifier
            for value in requirements
        ):
            continue
        entry = f"{module}:{BASES[kind].__name__}"
        options[entry] = f"{name} · {dist.version}"
    return dict(sorted(options.items()))


def resolve_algo(kind: str, entry: str) -> type:
    if entry not in algo_options(kind):
        raise ValueError(f"{kind} 上游未安装或 Scheme 大版本不兼容：{entry}")
    module, name = entry.split(":")
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, BASES[kind]):
        raise TypeError(f"{entry} 必须继承 {BASES[kind].__name__}")
    return cls


def research_components(
    params: BacktestParameters,
    kind: str,
    current: type[Algo[Any, Any]],
) -> dict[str, type[Algo[Any, Any]]]:
    """仅加载当前环节之前的选择；未提供的后续环节交由 Strategy 补默认值。"""
    base = BASES[kind]
    if not isinstance(current, type) or not issubclass(current, base):
        raise TypeError(f"run 需要 {base.__name__} 类")
    components = {}
    for upstream in list(BASES)[: list(BASES).index(kind)]:
        components[upstream] = resolve_algo(upstream, getattr(params, upstream))
    components[kind] = current
    return components


def run_research(
    params: BacktestParameters,
    kind: str,
    current: type[Algo[Any, Any]],
    *,
    ctx: ResearchContext[Any] | None = None,
    settings: "DolphinSettings | None" = None,
) -> "BacktestResult":
    from .api import run_backtest
    from .assembly import Strategy, context_type

    components = research_components(params, kind, current)
    context = ctx if ctx is not None else context_type(components["model"])()
    strategy = Strategy(context, params=params, **components)
    return run_backtest(strategy.algos, strategy.ctx, params, settings=settings)
