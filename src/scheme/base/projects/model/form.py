"""Model 研究表单：回测设置与默认后续算法。"""

from typing import Generic, Literal, TypeVar

from pydantic import Field

from scheme.base.internal.form import BacktestForm

from .params import ModelAnalysisParams

__all__ = ["ModelReportForm"]

P = TypeVar("P", bound=ModelAnalysisParams, infer_variance=True)


class ModelReportForm(BacktestForm[P], Generic[P]):
    analysis_base = ModelAnalysisParams
    bound_analysis = ModelAnalysisParams

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
