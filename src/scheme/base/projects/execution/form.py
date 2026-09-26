"""Execution 项目的回测报告表单。"""

from scheme.base.internal.backtest_form import BacktestForm
from scheme.base.internal.form import ReportForm

from .params import ExecutionAnalysisParams, ExecutionComponents

__all__ = ["ExecutionReportForm"]


class ExecutionReportForm(ExecutionComponents, BacktestForm, ReportForm[ExecutionAnalysisParams]):
    def build(self) -> ExecutionAnalysisParams:
        return ExecutionAnalysisParams(**self.backtest_values())
