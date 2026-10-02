"""因子运行参数及其分析参数。"""

from datetime import date
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scheme.base.internal.params import _all_market, validate_research
from scheme.data.dolphindb.symbols import engine_symbol
from scheme.data.dolphindb.universe import Universe

if TYPE_CHECKING:
    from scheme.config import DolphinSettings
    from scheme.execute.factor.result import FactorAnalysisResult

    from .algo import Factor

__all__ = ["FactorParams", "FactorAnalysisParams"]


class FactorParams(BaseModel):
    """构造因子所需的参数，不包含报告分析选项。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（不含）")
    universe: Universe = Field(default_factory=_all_market, title="股票池设置")

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        validate_research(self.start, self.end, self.universe)
        return self


class FactorAnalysisParams(FactorParams):
    """因子参数加评价设置；填写后通过 run(Factor) 生成报告。"""

    columns: list[str] = Field(min_length=1, title="因子列")
    return_periods: list[int] = Field(
        default_factory=lambda: [1, 5, 20], min_length=1, title="收益持有期"
    )
    groups: int = Field(default=5, ge=2, title="分组数量")
    n_select: int = Field(default=10, ge=1, title="极端股票数")
    weight: Literal["equal", "market_value"] = Field(
        default="equal", title="加权方式",
        json_schema_extra={"x-enum-labels": ["等权", "市值加权"]},
    )
    calendar_symbol: str = Field(default="000300.XSHG", title="交易日历代码")

    @model_validator(mode="after")
    def validate_analysis(self) -> Self:
        self.calendar_symbol = engine_symbol(self.calendar_symbol)
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("因子列不能重复")
        if any(period < 1 for period in self.return_periods):
            raise ValueError("收益持有期必须为正整数")
        if len(set(self.return_periods)) != len(self.return_periods):
            raise ValueError("收益持有期不能重复")
        return self

    def run[P: FactorParams](
        self, factor: "type[Factor[P]]", *, settings: "DolphinSettings | None" = None
    ) -> "FactorAnalysisResult[Self]":
        from scheme.execute.factor.api import analyze_factors
        from scheme.execute.strategy.assembly import parameter_type, project_parameters

        from .algo import Factor

        if not isinstance(factor, type) or not issubclass(factor, Factor):
            raise TypeError("run 需要 Factor 类")
        params = project_parameters(parameter_type(factor), self)
        return analyze_factors(factor(params), self, settings=settings)
