from collections.abc import Sequence

from pydantic import BaseModel, Field

from .models import Order, Signal, TargetPortfolio

__all__ = ["ResearchContext"]


class ResearchContext[T](BaseModel):
    """T 由 Model 定义；环节间的结果消费后清空，长期状态由算法自身保存。"""

    signal: Signal[T] | None = None
    target: TargetPortfolio | None = None
    orders: list[Order] = Field(default_factory=list)
    risk_decisions: Sequence[bool] = Field(default_factory=list)
    accepted_orders: list[Order] = Field(default_factory=list)
