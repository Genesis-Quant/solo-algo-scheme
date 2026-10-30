"""Solo 查询在 Runtime DSL 校验之上扩展前视计算区间。"""

from datetime import timedelta
from typing import Any

from pydantic import Field, TypeAdapter, field_validator
from runtime.apps.query import FactorQuery as RuntimeFactorQuery

__all__ = ["FactorQuery"]


class FactorQuery(RuntimeFactorQuery):
    lookforward: timedelta = Field(
        default=timedelta(0),
        ge=timedelta(0),
        description="计算时额外加载的未来日历时长；数据和日历仍只返回 start_date 至 end_date。",
        examples=["P20D"],
    )

    @field_validator("lookforward", mode="before")
    @classmethod
    def parse_lookforward(cls, value: Any) -> timedelta:
        if isinstance(value, timedelta):
            result = value
        elif isinstance(value, str):
            try:
                result = TypeAdapter(timedelta).validate_python(value)
            except ValueError as error:
                raise ValueError(f"lookforward 不是有效 TimeDelta：{value!r}") from error
        else:
            raise ValueError("lookforward 必须是 timedelta 或 TimeDelta 字符串")
        if result < timedelta(0):
            raise ValueError("lookforward 不能小于 0")
        return result
