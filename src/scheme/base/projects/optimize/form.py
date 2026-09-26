"""Optimize 项目的回测报告表单。"""

from scheme.base.internal.backtest_form import BacktestForm
from scheme.base.internal.form import ReportForm

from .params import OptimizeAnalysisParams, OptimizeComponents

__all__ = ["OptimizeReportForm"]


class OptimizeReportForm(OptimizeComponents, BacktestForm, ReportForm[OptimizeAnalysisParams]):
    def build(self) -> OptimizeAnalysisParams:
        return OptimizeAnalysisParams(**self.backtest_values())
