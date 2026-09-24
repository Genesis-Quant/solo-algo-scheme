"""策略组装契约、默认算法和环节顺序。"""

from functools import cached_property
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .algo import Algo, ControlAlgo, ExecutionAlgo, OptimizeAlgo
from .context import ResearchContext
from .params import StrategyParams, parameter_type

__all__ = ["AlgoComponents", "Strategy"]


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
        from .defaults import DirectExecution, NoControl, RiskParity

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
        from .utils.packages import validate_context

        for algo in self.algos:
            validate_context(algo, ctx)

    @cached_property
    def universe(self) -> pd.DataFrame:
        """参数区间的股票池 bool 面板，首次访问时查询。"""
        return self.params.universe.evaluate(self.params.start, self.params.end)
