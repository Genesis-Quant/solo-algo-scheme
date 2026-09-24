from pydantic import BaseModel, ConfigDict

from .models import Order, Signal, Target

__all__ = ["ResearchContext"]


class ResearchContext[T](BaseModel):
    """T 由 Model 定义；环节间的结果消费后清空，长期状态由算法自身保存。"""

    model_config = ConfigDict(validate_assignment=True, allow_inf_nan=False)

    signal: Signal[T] | None = None  # Model->Optimize
    target: Target | None = None  # Optimize->Control，目标资产权重；正数做多，负数做空
    orders: list[Order] | None = None  # Control->Execution
