"""Model 项目的算法参数与回测分析参数。"""

from datetime import date
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scheme.base.internal.context import ResearchContext
from scheme.base.internal.params import _all_market, validate_backtest, validate_research
from scheme.data.dolphindb.universe import Universe

if TYPE_CHECKING:
    from scheme.config import DolphinSettings
    from scheme.execute.strategy.result import BacktestResult

    from .algo import ModelAlgo

__all__ = ["ModelParams", "ModelAnalysisParams"]


class ModelParams(BaseModel):
    """从统一策略参数中读取当前算法声明的字段。"""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class ModelAnalysisParams(ModelParams):
    """完整策略研究参数；将 Model 和选中的后续 Algo 组装后运行。"""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（不含）")
    universe: Universe = Field(default_factory=_all_market, title="股票池设置")
    market_data: Literal["stock_daily", "stock_snapshot"] = Field(
        default="stock_daily",
        title="行情类型",
        json_schema_extra={"x-enum-labels": ["日频合成快照", "Tick 快照"]},
    )
    symbols: list[str] | None = Field(default=None, min_length=1, title="行情股票范围")
    benchmark: str | None = Field(default=None, title="基准代码")
    batch_days: int = Field(default=1, ge=1, title="行情加载批次天数")
    config: dict[str, Any] = Field(
        default_factory=lambda: {"cash": 1_000_000.0}, title="回测引擎配置"
    )
    optimize: Literal["risk_parity"] = Field(
        default="risk_parity", title="组合优化", json_schema_extra={"x-enum-labels": ["风险平价"]}
    )
    control: Literal["no_control"] = Field(
        default="no_control", title="订单风控", json_schema_extra={"x-enum-labels": ["不风控"]}
    )
    execution: Literal["direct_execution"] = Field(
        default="direct_execution",
        title="算法下单",
        json_schema_extra={"x-enum-labels": ["不拆单"]},
    )

    @model_validator(mode="after")
    def validate_analysis(self) -> Self:
        validate_research(self.start, self.end, self.universe)
        self.symbols, self.benchmark = validate_backtest(self.config, self.symbols, self.benchmark)
        return self

    def run[P: ModelParams, C: ResearchContext[Any]](
        self,
        algo: "type[ModelAlgo[P, C]]",
        *,
        ctx: C | None = None,
        settings: "DolphinSettings | None" = None,
    ) -> "BacktestResult[Self]":
        from scheme.execute.strategy.components import run_research

        return run_research(self, "model", algo, ctx=ctx, settings=settings)
