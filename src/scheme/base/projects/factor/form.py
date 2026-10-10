"""用于 UI 的因子分析表单。"""

from typing import Literal

from pydantic import Field

from scheme.base.internal.form import ResearchForm

from .params import FactorAnalysisParams

__all__ = ["FactorReportForm"]


class FactorReportForm[P: FactorAnalysisParams](ResearchForm[P]):
    _analysis_base = FactorAnalysisParams
    _analysis_model = FactorAnalysisParams

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
