from collections.abc import Sequence

from pydantic import BaseModel, Field

from .models import Order, Recommendation, TargetPortfolio

__all__ = ["ResearchContext"]


class ResearchContext(BaseModel):
    """环节间待消费的结果；消费者取出后清空，长期状态由算法自身保存。"""

    recommend: Recommendation | None = None
    target: TargetPortfolio | None = None
    orders: list[Order] = Field(default_factory=list)
    risk_decisions: Sequence[bool] = Field(default_factory=list)
    accepted_orders: list[Order] = Field(default_factory=list)
