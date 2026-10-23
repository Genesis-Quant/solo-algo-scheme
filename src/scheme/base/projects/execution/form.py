"""Execution 项目的回测报告表单。"""

from typing import Generic, TypeVar

from pydantic import Field

from scheme.base.internal.form import BacktestForm

from .params import ExecutionAnalysisParams

__all__ = ["ExecutionReportForm"]

P = TypeVar("P", bound=ExecutionAnalysisParams, infer_variance=True)


class ExecutionReportForm(BacktestForm[P], Generic[P]):
    analysis_base = ExecutionAnalysisParams
    bound_analysis = ExecutionAnalysisParams

    model: str = Field(min_length=1, title="策略建模", json_schema_extra={"x-algo-kind": "model"})
    optimize: str = Field(
        min_length=1, title="组合优化", json_schema_extra={"x-algo-kind": "optimize"}
    )
    control: str = Field(
        min_length=1, title="订单风控", json_schema_extra={"x-algo-kind": "control"}
    )
