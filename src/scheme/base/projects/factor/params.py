"""因子运行参数及其分析参数。"""

from datetime import date
from typing import TYPE_CHECKING, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scheme.data.dolphindb.symbols import engine_symbol
from scheme.data.dolphindb.universe import Universe

if TYPE_CHECKING:
    from scheme.config import DolphinSettings
    from scheme.execute.factor.result import FactorAnalysisResult

    from .algo import Factor

__all__ = ["FactorParams", "FactorAnalysisParams"]

P = TypeVar("P", bound="FactorParams")


class FactorParams(BaseModel):
    """构造因子所需的股票池及算法参数，不包含日期或报告分析选项。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    universe: Universe = Field(default_factory=Universe, title="股票池设置")


class FactorAnalysisParams(FactorParams):
    """因子参数加闭区间日期及评价设置；通过 run(Factor) 生成报告。"""

    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（含）")
    columns: list[str] = Field(default_factory=list, title="因子列（内部运行与报告元数据）")
    period: int = Field(default=1, ge=1, strict=True, title="持有期（交易日）")
    price_column: Literal["open", "close", "open_hfq", "close_hfq"] = Field(
        default="open_hfq", title="收益价格字段",
    )
    max_lag: int = Field(default=1, ge=1, strict=True, title="最大滞后阶数（交易日）")
    groups: int = Field(default=5, ge=2, title="分组数量")
    n_select: int = Field(default=10, ge=1, title="极端股票数")
    weight: Literal["equal", "market_value"] = Field(
        default="equal", title="加权方式",
        json_schema_extra={"x-enum-labels": ["等权", "市值加权"]},
    )
    calendar_symbol: str = Field(default="000300.XSHG", title="交易日历代码")

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        if self.start > self.end:
            raise ValueError("日期区间必须满足 start <= end，采用 [start, end]")
        return self

    @model_validator(mode="after")
    def validate_analysis(self) -> Self:
        self.calendar_symbol = engine_symbol(self.calendar_symbol)
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("因子列不能重复")
        reserved = {
            "date", "symbol", "time", "code", "stock_pool_member", "circ_mv", "industry_l0",
            *self.return_columns,
        }
        if set(self.columns) & reserved:
            raise ValueError("因子列不能与收益列或分析保留列重名")
        return self

    @property
    def return_columns(self) -> list[str]:
        return [f"P{self.period}L{lag}" for lag in range(1, self.max_lag + 1)]

    @property
    def return_specs(self) -> dict[str, dict[str, str | int]]:
        return {
            name: {
                "kind": "log", "periods": self.period, "lag": lag,
                "price_column": self.price_column, "turnover_period": self.period,
            }
            for lag, name in enumerate(self.return_columns, start=1)
        }

    def _bind_columns(self, factor: "type[Factor]") -> None:
        """类声明是运行与报告的唯一列来源；不修改类上的可变列表。"""
        columns = list(factor.columns)
        # 绑定时也验证按本次 period/max_lag 动态生成的收益列冲突。
        self.model_copy(update={"columns": columns}).validate_analysis()
        self.columns = columns

    def run(
        self, factor: "type[Factor[P]]", *, settings: "DolphinSettings | None" = None
    ) -> "FactorAnalysisResult[Self]":
        from scheme.execute.factor.api import analyze_factors
        from scheme.execute.strategy.assembly import parameter_type, project_parameters

        from .algo import Factor

        if not isinstance(factor, type) or not issubclass(factor, Factor):
            raise TypeError("run 需要 Factor 类")
        self._bind_columns(factor)
        params = project_parameters(parameter_type(factor), self)
        return analyze_factors(factor(params), self, settings=settings)
