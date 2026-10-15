"""ExecutionAlgo 研究项目基类。"""

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from scheme.base.internal.algo import Algo
from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Order

__all__ = ["ExecutionAlgo"]

P = TypeVar("P", bound=BaseModel, infer_variance=True)
C = TypeVar("C", bound=ResearchContext[Any], infer_variance=True)


class ExecutionAlgo(Algo[P, C], Generic[P, C]):
    def process(self) -> bool:
        orders = self.ctx.orders
        if orders is None:
            return False
        self.ctx.orders = None
        self.on_orders(orders)
        return True

    @abstractmethod
    def on_orders(self, orders: list[Order]) -> None:
        """根据已消费的 orders 执行拆单和下单。"""
        ...
