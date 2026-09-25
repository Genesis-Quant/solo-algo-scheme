"""面向插件和 Notebook 的预设选项；build 后得到完整分析参数。"""

from datetime import timedelta
from typing import Literal

from pydantic import Field

from scheme.base.internal.form import ReportForm
from scheme.data.dolphindb.universe import StockPool, Universe

from .params import AnalysisSettings, FactorAnalysisParams

__all__ = ["FactorReportForm"]


class FactorReportForm(AnalysisSettings, ReportForm[FactorAnalysisParams]):
    """预设分析选项；插件根据该 Pydantic 模型生成输入表单。"""

    pool: Literal[
        StockPool.ALL, StockPool.SSE50, StockPool.CSI300, StockPool.CSI500, StockPool.CSI1000
    ] = Field(default=StockPool.ALL, title="股票池")
    lookback: timedelta = Field(default=timedelta(0), title="回溯周期")

    def build(self) -> FactorAnalysisParams:
        """把预设股票池展开为可独立运行的分析参数。"""
        return FactorAnalysisParams(
            **self.model_dump(exclude={"pool", "lookback"}),
            universe=Universe(pool=self.pool, lookback=self.lookback),
        )
