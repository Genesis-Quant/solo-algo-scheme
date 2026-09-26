"""Control 项目的算法参数基类。"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scheme.base.internal.params import BacktestParameters

if TYPE_CHECKING:
    from scheme.base.internal.context import ResearchContext
    from scheme.config import DolphinSettings
    from scheme.execute.strategy.result import BacktestResult

    from .algo import ControlAlgo

__all__ = ["ControlParams", "ControlAnalysisParams"]


class ControlParams(BaseModel):
    """从统一策略参数中读取当前算法声明的字段。"""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class ControlComponents(BaseModel):
    model: str = Field(min_length=1, title="策略建模", json_schema_extra={"x-algo-kind": "model"})
    optimize: str = Field(
        min_length=1, title="组合优化", json_schema_extra={"x-algo-kind": "optimize"}
    )
    execution: Literal["direct_execution"] = Field(
        default="direct_execution",
        title="算法下单",
        json_schema_extra={"x-enum-labels": ["不拆单"]},
    )


class ControlAnalysisParams(BacktestParameters, ControlComponents):
    """使用选定上游、当前算法与默认后续环节生成回测报告。"""

    def run(
        self,
        algo: "type[ControlAlgo[Any, Any]]",
        *,
        ctx: "ResearchContext[Any] | None" = None,
        settings: "DolphinSettings | None" = None,
    ) -> "BacktestResult":
        from scheme.execute.strategy.components import run_research

        return run_research(self, "control", algo, ctx=ctx, settings=settings)
