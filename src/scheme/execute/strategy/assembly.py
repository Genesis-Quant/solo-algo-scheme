"""策略组装契约、默认算法和环节顺序。"""

from functools import cached_property
from typing import Any, get_args, get_origin

import pandas as pd
from pydantic import BaseModel, ConfigDict

from scheme.base import (
    Algo,
    ControlAlgo,
    ExecutionAlgo,
    OptimizeAlgo,
    ResearchContext,
    StrategyParams,
)

__all__ = ["AlgoComponents", "Strategy"]


def _generic_model(
    cls: type, index: int, bindings: dict[Any, Any] | None = None
) -> type[BaseModel]:
    """从 Factor[P] / Algo[P, C] 的继承关系解析参数或上下文模型。"""
    bindings = bindings or {}
    for base in cls.__dict__.get("__orig_bases__", cls.__bases__):
        origin = get_origin(base) or base
        args = tuple(bindings.get(arg, arg) for arg in get_args(base))
        if origin.__name__ in {"Algo", "Factor"} and origin.__module__.split(".")[0] in {
            "backtest",
            "scheme",
        }:
            if len(args) > index:
                model = args[index]
                model = getattr(model, "__bound__", None) or model
                if isinstance(model, type) and issubclass(model, BaseModel):
                    return model
        mapping = dict(zip(getattr(origin, "__type_params__", ()), args))
        if origin is not object:
            try:
                return _generic_model(origin, index, mapping)
            except TypeError:
                pass
    raise TypeError(f"{cls.__name__} 必须明确指定 BaseModel 参数泛型")


def parameter_type(cls: type) -> type[BaseModel]:
    return _generic_model(cls, 0)


def context_type(cls: type) -> type[ResearchContext[Any]]:
    context = _generic_model(cls, 1)
    if not issubclass(context, ResearchContext):
        raise TypeError("Model 上下文必须继承 ResearchContext")
    return context


def validate_context(algo: Any, ctx: BaseModel) -> None:
    expected = _generic_model(type(algo), 1)
    generic = getattr(expected, "__pydantic_generic_metadata__", {})
    if generic.get("args") == (Any,):
        expected = generic["origin"]
    if not isinstance(ctx, expected):
        raise TypeError(
            f"{type(algo).__name__} 需要 {expected.__name__}，收到 {type(ctx).__name__}"
        )


class AlgoComponents[T](BaseModel):
    """T 可以是待安装的包描述；字段与顺序由 scheme 维护。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: T
    optimize: T | None = None
    control: T | None = None
    execution: T | None = None


class Strategy[C: ResearchContext[Any]]:
    """从统一参数实例化 Algo 类，省略的下游环节使用默认算法。"""

    def __init__(
        self,
        ctx: C,
        *,
        params: StrategyParams,
        model: type[Algo[Any, C]],
        optimize: type[OptimizeAlgo[Any, C]] | None = None,
        control: type[ControlAlgo[Any, C]] | None = None,
        execution: type[ExecutionAlgo[Any, C]] | None = None,
    ) -> None:
        from scheme.base.projects.control.default import NoControl
        from scheme.base.projects.execution.default import DirectExecution
        from scheme.base.projects.optimize.default import RiskParity

        self.ctx = ctx
        self.params = params
        components = (
            (model, Algo),
            (optimize if optimize is not None else RiskParity, OptimizeAlgo),
            (control if control is not None else NoControl, ControlAlgo),
            (execution if execution is not None else DirectExecution, ExecutionAlgo),
        )
        for cls, base in components:
            if not isinstance(cls, type) or not issubclass(cls, base):
                raise TypeError(f"{base.__name__} 环节需要传入对应的 Algo 类")
        self.algos: tuple[Algo[Any, C], ...] = tuple(
            cls(parameter_type(cls).model_validate(params, from_attributes=True))
            for cls, _ in components
        )
        for algo in self.algos:
            validate_context(algo, ctx)

    @cached_property
    def universe(self) -> pd.DataFrame:
        """参数区间的股票池 bool 面板，首次访问时查询。"""
        return self.params.universe.evaluate(self.params.start, self.params.end)
