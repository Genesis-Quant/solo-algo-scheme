"""ExecutionAlgo 研究项目基类。"""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from scheme.base.internal.algo import Algo
from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Order
from scheme.data.dolphindb.symbols import normalize_order

__all__ = ["ExecutionAlgo"]


class ExecutionAlgo[P: BaseModel, C: ResearchContext[Any]](Algo[P, C]):
    def process(self) -> bool:
        orders = self.ctx.orders
        if orders is None:
            return False
        orders = [normalize_order(order) for order in orders]
        self.ctx.orders = None
        self.on_orders(orders)
        return True

    @abstractmethod
    def on_orders(self, orders: list[Order]) -> None:
        """根据已消费的 orders 执行拆单和下单。"""
        ...
