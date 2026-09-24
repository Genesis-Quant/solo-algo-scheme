"""Factor 和 Strategy 共用的日期与股票池参数。"""

from datetime import date
from typing import Any, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .universe import Universe

__all__ = ["FactorParams", "StrategyParams", "generic_model", "parameter_type"]


def _all_market() -> Universe:
    return Universe(
        derivatives={"member": {"type": "DIRECT", "op": "nullary.true", "fields": {}}},
        filters=["member"],
    )


class FactorParams(BaseModel):
    """因子项目参数基类。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start: date
    end: date
    universe: Universe = Field(default_factory=_all_market)

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        if self.start >= self.end:
            raise ValueError("日期区间必须满足 start < end，采用 [start, end)")
        self.universe.query(self.start, self.end)
        return self


class StrategyParams(BaseModel):
    """策略统一参数；额外字段保留，供所有 Algo 参数模型读取。"""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    start: date
    end: date
    universe: Universe = Field(default_factory=_all_market)

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        if self.start >= self.end:
            raise ValueError("日期区间必须满足 start < end，采用 [start, end)")
        self.universe.query(self.start, self.end)
        return self


def generic_model(cls: type, index: int, bindings: dict[Any, Any] | None = None) -> type[BaseModel]:
    """从 Factor[P] / Algo[P, C] 的继承关系解析参数或上下文模型。"""
    bindings = bindings or {}
    for base in cls.__dict__.get("__orig_bases__", cls.__bases__):
        origin = get_origin(base) or base
        args = tuple(bindings.get(arg, arg) for arg in get_args(base))
        if origin.__name__ in {"Algo", "Factor"} and origin.__module__.split(".")[0] in {
            "backtest",
            "scheme",
        }:
            if len(args) > index:
                model = args[index]
                model = getattr(model, "__bound__", None) or model
                if isinstance(model, type) and issubclass(model, BaseModel):
                    return model
        mapping = dict(zip(getattr(origin, "__type_params__", ()), args))
        if origin is not object:
            try:
                return generic_model(origin, index, mapping)
            except TypeError:
                pass
    raise TypeError(f"{cls.__name__} 必须明确指定 BaseModel 参数泛型")


def parameter_type(cls: type) -> type[BaseModel]:
    return generic_model(cls, 0)
