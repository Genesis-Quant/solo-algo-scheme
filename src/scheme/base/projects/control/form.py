"""Control 项目的回测报告表单。"""

from scheme.base.internal.backtest_form import BacktestForm
from scheme.base.internal.form import ReportForm

from .params import ControlAnalysisParams, ControlComponents

__all__ = ["ControlReportForm"]


class ControlReportForm(ControlComponents, BacktestForm, ReportForm[ControlAnalysisParams]):
    def build(self) -> ControlAnalysisParams:
        return ControlAnalysisParams(**self.backtest_values())
