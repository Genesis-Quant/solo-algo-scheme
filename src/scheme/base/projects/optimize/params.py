"""Optimize 项目的算法参数基类。"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scheme.base.internal.params import BacktestParameters

if TYPE_CHECKING:
    from scheme.base.internal.context import ResearchContext
    from scheme.config import DolphinSettings
    from scheme.execute.strategy.result import BacktestResult

    from .algo import OptimizeAlgo

__all__ = ["OptimizeParams", "OptimizeAnalysisParams"]


class OptimizeParams(BaseModel):
    """从统一策略参数中读取当前算法声明的字段。"""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class OptimizeComponents(BaseModel):
    model: str = Field(min_length=1, title="策略建模", json_schema_extra={"x-algo-kind": "model"})
    control: Literal["no_control"] = Field(
        default="no_control", title="订单风控", json_schema_extra={"x-enum-labels": ["不风控"]}
    )
    execution: Literal["direct_execution"] = Field(
        default="direct_execution",
        title="算法下单",
        json_schema_extra={"x-enum-labels": ["不拆单"]},
    )


class OptimizeAnalysisParams(BacktestParameters, OptimizeComponents):
    """使用选定上游、当前算法与默认后续环节生成回测报告。"""

    def run(
        self,
        algo: "type[OptimizeAlgo[Any, Any]]",
        *,
        ctx: "ResearchContext[Any] | None" = None,
        settings: "DolphinSettings | None" = None,
    ) -> "BacktestResult":
        from scheme.execute.strategy.components import run_research

        return run_research(self, "optimize", algo, ctx=ctx, settings=settings)
