"""用于 UI 的因子分析表单。"""

from datetime import date
from typing import Generic, Literal, TypeVar

from pydantic import Field

from scheme.base.internal.form import ResearchForm

from .params import FactorAnalysisParams

__all__ = ["FactorReportForm"]

P = TypeVar("P", bound=FactorAnalysisParams, infer_variance=True)


class FactorReportForm(ResearchForm[P], Generic[P]):
    _analysis_base = FactorAnalysisParams
    _analysis_model = FactorAnalysisParams

    end: date = Field(default=date(2027, 1, 1), title="结束日期（含）")
    columns: list[str] = Field(min_length=1, title="因子列")
    period: int = Field(default=1, ge=1, strict=True, title="持有期（交易日）")
    price_column: Literal["open", "close", "open_hfq", "close_hfq"] = Field(
        default="open_hfq", title="收益价格字段",
    )
    max_lag: int = Field(default=1, ge=1, strict=True, title="最大滞后阶数（交易日）")
    groups: int = Field(default=5, ge=2, title="分组数量")
    n_select: int = Field(default=10, ge=1, title="极端股票数")
    weight: Literal["equal", "market_value"] = Field(
        default="market_value", title="加权方式",
        json_schema_extra={"x-enum-labels": ["等权", "市值加权"]},
    )
    calendar_symbol: str = Field(default="000300.XSHG", title="交易日历代码")
