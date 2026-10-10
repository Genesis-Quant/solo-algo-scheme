"""Optimize 项目的回测报告表单。"""

from typing import Literal

from pydantic import Field

from scheme.base.internal.form import BacktestForm

from .params import OptimizeAnalysisParams

__all__ = ["OptimizeReportForm"]


class OptimizeReportForm[P: OptimizeAnalysisParams](BacktestForm[P]):
    _analysis_base = OptimizeAnalysisParams
    _analysis_model = OptimizeAnalysisParams

    model: str = Field(min_length=1, title="策略建模", json_schema_extra={"x-algo-kind": "model"})
    control: Literal["no_control"] = Field(
        default="no_control", title="订单风控", json_schema_extra={"x-enum-labels": ["不风控"]}
    )
    execution: Literal["direct_execution"] = Field(
        default="direct_execution", title="算法下单",
        json_schema_extra={"x-enum-labels": ["不拆单"]},
    )
