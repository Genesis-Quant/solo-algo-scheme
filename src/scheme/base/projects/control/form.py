"""Control 项目的回测报告表单。"""

from typing import Literal

from pydantic import Field

from scheme.base.internal.form import BacktestForm

from .params import ControlAnalysisParams

__all__ = ["ControlReportForm"]


class ControlReportForm[P: ControlAnalysisParams](BacktestForm[P]):
    _analysis_base = ControlAnalysisParams
    _analysis_model = ControlAnalysisParams

    model: str = Field(min_length=1, title="策略建模", json_schema_extra={"x-algo-kind": "model"})
    optimize: str = Field(
        min_length=1, title="组合优化", json_schema_extra={"x-algo-kind": "optimize"}
    )
    execution: Literal["direct_execution"] = Field(
        default="direct_execution", title="算法下单",
        json_schema_extra={"x-enum-labels": ["不拆单"]},
    )
