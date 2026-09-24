from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from .models import Order, Signal, Target
from .symbols import normalize_assets, normalize_order

__all__ = ["ResearchContext"]


class ResearchContext[T](BaseModel):
    """T 由 Model 定义；环节间的结果消费后清空，长期状态由算法自身保存。"""

    model_config = ConfigDict(validate_assignment=True, allow_inf_nan=False)

    signal: Signal[T] | None = None  # Model->Optimize
    target: Target | None = None  # Optimize->Control，目标资产权重；正数做多，负数做空
    orders: list[Order] | None = None  # Control->Execution

    @field_validator("signal", "target")
    @classmethod
    def normalize_messages(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return None if value is None else normalize_assets(value)

    @field_validator("orders")
    @classmethod
    def normalize_orders(cls, value: list[Order] | None) -> list[Order] | None:
        return None if value is None else [normalize_order(order) for order in value]
