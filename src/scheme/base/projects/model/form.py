"""Model 研究表单：回测设置与默认后续算法。"""

from scheme.base.internal.backtest_form import BacktestForm
from scheme.base.internal.form import ReportForm

from .params import DefaultAlgos, ModelAnalysisParams

__all__ = ["ModelReportForm"]


class ModelReportForm(DefaultAlgos, BacktestForm, ReportForm[ModelAnalysisParams]):
    def build(self) -> ModelAnalysisParams:
        return ModelAnalysisParams(**self.backtest_values())
