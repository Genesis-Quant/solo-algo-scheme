"""Model 项目的算法参数基类。"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scheme.base.internal.params import BacktestParameters

if TYPE_CHECKING:
    from scheme.base.internal.context import ResearchContext
    from scheme.config import DolphinSettings
    from scheme.execute.strategy.result import BacktestResult

    from .algo import ModelAlgo

__all__ = ["ModelParams", "ModelAnalysisParams"]


class ModelParams(BaseModel):
    """从统一策略参数中读取当前算法声明的字段。"""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class DefaultAlgos(BaseModel):
    """策略研究可选的 Scheme 内置后续算法。"""

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

    def default_algos(self) -> dict[str, type]:
        from scheme.base.projects.control.default import NoControl
        from scheme.base.projects.execution.default import DirectExecution
        from scheme.base.projects.optimize.default import RiskParity

        return {
            "optimize": {"risk_parity": RiskParity}[self.optimize],
            "control": {"no_control": NoControl}[self.control],
            "execution": {"direct_execution": DirectExecution}[self.execution],
        }


class ModelAnalysisParams(BacktestParameters, DefaultAlgos):
    """完整策略研究参数；将 Model 和选中的后续 Algo 组装后运行。"""

    def run(
        self,
        model: "type[ModelAlgo[Any, Any]]",
        *,
        ctx: "ResearchContext[Any] | None" = None,
        settings: "DolphinSettings | None" = None,
    ) -> "BacktestResult":
        from scheme.execute.strategy.api import run_backtest
        from scheme.execute.strategy.assembly import Strategy, context_type

        from .algo import ModelAlgo

        if not isinstance(model, type) or not issubclass(model, ModelAlgo):
            raise TypeError("run 需要 ModelAlgo 类")
        context = ctx if ctx is not None else context_type(model)()
        strategy = Strategy(context, params=self, model=model, **self.default_algos())
        return run_backtest(strategy.algos, strategy.ctx, self, settings=settings)
