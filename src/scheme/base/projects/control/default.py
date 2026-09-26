"""control 项目的默认实现。"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Order, Target
from scheme.base.projects.control import ControlAlgo
from scheme.base.projects.control import ControlParams as BaseControlParams

__all__ = ["ControlParams", "NoControl"]


class ControlParams(BaseControlParams):
    lot_size: int = Field(default=100, ge=1)


class NoControl[C: ResearchContext[Any]](ControlAlgo[ControlParams, C]):
    def __init__(self, params: ControlParams | None = None) -> None:
        super().__init__(params if params is not None else ControlParams())

    def on_target(self, target: Target) -> None:
        self.ctx.orders = self.target_orders(target)

    def target_orders(self, target: Target) -> list[Order]:
        return super().target_orders(target, lot_size=self.params.lot_size)
