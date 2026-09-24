from typing import Literal

from pydantic import Field, model_validator

from scheme import FactorParams
from scheme.utils.schema import Component, Task


class FactorAnalysisParameters(FactorParams):
    columns: list[str] = Field(min_length=1)
    return_periods: list[int] = Field(default_factory=lambda: [1, 5, 20], min_length=1)
    groups: int = Field(default=5, ge=2)
    n_select: int = Field(default=10, ge=1)
    weight: Literal["equal", "market_value"] = "equal"
    calendar_symbol: str = "000300.XSHG"

    @model_validator(mode="after")
    def check_columns(self) -> "FactorAnalysisParameters":
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("因子列不能重复")
        if any(period < 1 for period in self.return_periods):
            raise ValueError("收益持有期必须为正整数")
        if len(set(self.return_periods)) != len(self.return_periods):
            raise ValueError("收益持有期不能重复")
        return self


class FactorTask(Task):
    kind: Literal["factor"] = "factor"
    factor: Component
    analysis: FactorAnalysisParameters
