from pydantic import BaseModel, Field

from .models import Order, Signal, Target

__all__ = ["ResearchContext"]


class ResearchContext[T](BaseModel):
    """T 由 Model 定义；环节间的结果消费后清空，长期状态由算法自身保存。"""

    signal: Signal[T] | None = None  # Model->Optimize
    target: Target | None = None  # Optimize->Execution
    orders: list[Order] = Field(default_factory=list)  # Execution->Control
