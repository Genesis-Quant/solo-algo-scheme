"""execution 项目的默认实现。"""

from __future__ import annotations

from datetime import time
from typing import Any, Generic, TypeVar

import pandas as pd

from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Order
from scheme.base.projects.execution import ExecutionAlgo
from scheme.base.projects.execution import ExecutionParams as BaseExecutionParams
from scheme.log import get_logger

__all__ = ["DefaultParams", "DirectExecution"]

log = get_logger("scheme.execution")

C = TypeVar("C", bound=ResearchContext[Any], infer_variance=True)


class DefaultParams(BaseExecutionParams):
    pass


class DirectExecution(ExecutionAlgo[DefaultParams, C], Generic[C]):
    def __init__(self, params: DefaultParams | None = None) -> None:
        super().__init__(params if params is not None else DefaultParams())

    def process(self) -> bool:
        now = pd.Timestamp(self.backtest.time).time()
        if not (time(9, 30) <= now < time(11, 30) or time(13) <= now < time(15)):
            return False
        return super().process()

    def on_orders(self, orders: list[Order]) -> None:
        now = pd.Timestamp(self.backtest.time).to_pydatetime()
        for pending in orders:
            order = pending.model_copy(update={"time": now})
            self.backtest.submit_order(order)
        log.debug("直接提交 {} 笔订单", len(orders))
