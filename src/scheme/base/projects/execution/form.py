"""Execution 项目的回测报告表单。"""

from pydantic import Field

from scheme.base.internal.form import BacktestForm

from .params import ExecutionAnalysisParams

__all__ = ["ExecutionReportForm"]


class ExecutionReportForm[P: ExecutionAnalysisParams](BacktestForm[P]):
    _analysis_base = ExecutionAnalysisParams
    _analysis_model = ExecutionAnalysisParams

    model: str = Field(min_length=1, title="策略建模", json_schema_extra={"x-algo-kind": "model"})
    optimize: str = Field(
        min_length=1, title="组合优化", json_schema_extra={"x-algo-kind": "optimize"}
    )
    control: str = Field(
        min_length=1, title="订单风控", json_schema_extra={"x-algo-kind": "control"}
    )
