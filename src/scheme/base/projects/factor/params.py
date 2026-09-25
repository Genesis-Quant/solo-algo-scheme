"""因子报告的完整分析参数与执行入口。"""

from datetime import date
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scheme.base.internal.params import _all_market
from scheme.data.dolphindb.symbols import engine_symbol
from scheme.data.dolphindb.universe import Universe

if TYPE_CHECKING:
    from scheme.config import DolphinSettings
    from scheme.execute.factor.result import FactorAnalysisResult

    from .algo import Factor

__all__ = ["FactorParams", "FactorAnalysisParams"]


class FactorParams(BaseModel):
    """因子项目参数基类。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（不含）")
    universe: Universe = Field(default_factory=_all_market, title="股票池设置")

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        if self.start >= self.end:
            raise ValueError("日期区间必须满足 start < end，采用 [start, end)")
        self.universe.query(self.start, self.end)
        return self


class AnalysisSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（不含）")
    columns: list[str] = Field(min_length=1, title="因子列")
    return_periods: list[int] = Field(
        default_factory=lambda: [1, 5, 20], min_length=1, title="收益持有期"
    )
    groups: int = Field(default=5, ge=2, title="分组数量")
    n_select: int = Field(default=10, ge=1, title="极端股票数")
    weight: Literal["equal", "market_value"] = Field(
        default="equal",
        title="加权方式",
        json_schema_extra={"x-enum-labels": ["等权", "市值加权"]},
    )
    calendar_symbol: str = Field(default="000300.XSHG", title="交易日历代码")

    @model_validator(mode="after")
    def check_columns(self) -> Self:
        if self.start >= self.end:
            raise ValueError("日期区间必须满足 start < end，采用 [start, end)")
        self.calendar_symbol = engine_symbol(self.calendar_symbol)
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("因子列不能重复")
        if any(period < 1 for period in self.return_periods):
            raise ValueError("收益持有期必须为正整数")
        if len(set(self.return_periods)) != len(self.return_periods):
            raise ValueError("收益持有期不能重复")
        return self


class FactorAnalysisParams(FactorParams, AnalysisSettings):
    """完整分析参数；可直接传入自定义 Universe，不依赖表单预设。"""

    @property
    def factor_params(self) -> FactorParams:
        """只将日期与股票池交给因子，避免将报告设置作为算法参数。"""
        return FactorParams.model_validate(self.model_dump(include={"start", "end", "universe"}))

    def run(
        self, factor: "type[Factor[FactorParams]]", *, settings: "DolphinSettings | None" = None
    ) -> "FactorAnalysisResult":
        """构造因子实例、计算因子并返回完整报告：params.run(Factor)。"""
        from scheme.execute.factor.api import analyze_factors

        from .algo import Factor

        if not isinstance(factor, type) or not issubclass(factor, Factor):
            raise TypeError("run 需要 Factor 类")
        return analyze_factors(factor(self.factor_params), self, settings=settings)
