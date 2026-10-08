"""用于 UI 的因子分析表单。"""

from datetime import date, timedelta
from typing import Literal

from pydantic import Field

from scheme.base.internal.form import ReportForm
from scheme.data.dolphindb.universe import StockPool, Universe

from .params import FactorAnalysisParams

__all__ = ["FactorReportForm"]


class FactorReportForm(ReportForm[FactorAnalysisParams]):
    start: date = Field(default=date(2020, 1, 1), title="开始日期")
    end: date = Field(default=date(2027, 1, 1), title="结束日期（不含）")
    pool: Literal[
        StockPool.ALL, StockPool.SSE50, StockPool.CSI300, StockPool.CSI500, StockPool.CSI1000
    ] = Field(default=StockPool.CSI300, title="股票池")
    lookback: timedelta = Field(default=timedelta(0), title="回溯周期")
    columns: list[str] = Field(min_length=1, title="因子列")
    return_periods: list[int] = Field(
        default_factory=lambda: [1, 5, 20], min_length=1, title="收益持有期"
    )
    groups: int = Field(default=5, ge=2, title="分组数量")
    n_select: int = Field(default=10, ge=1, title="极端股票数")
    weight: Literal["equal", "market_value"] = Field(
        default="market_value", title="加权方式",
        json_schema_extra={"x-enum-labels": ["等权", "市值加权"]},
    )
    calendar_symbol: str = Field(default="000300.XSHG", title="交易日历代码")

    def build(self) -> FactorAnalysisParams:
        return FactorAnalysisParams(
            **self.model_dump(exclude={"pool", "lookback"}),
            universe=Universe.model_validate({"pool": self.pool, "lookback": self.lookback}),
        )
