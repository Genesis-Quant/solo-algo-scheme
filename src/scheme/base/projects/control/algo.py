"""ControlAlgo 研究项目基类。"""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from scheme.base.internal.algo import Algo
from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Target
from scheme.data.dolphindb.symbols import normalize_assets

__all__ = ["ControlAlgo"]


class ControlAlgo[P: BaseModel, C: ResearchContext[Any]](Algo[P, C]):
    def process(self) -> bool:
        target = self.ctx.target
        if target is None or self.ctx.orders is not None:
            return False
        target = normalize_assets(target)
        self.ctx.target = None
        self.on_target(target)
        return True

    @abstractmethod
    def on_target(self, target: Target) -> None:
        """根据目标权重生成并检查订单，将通过的订单写入 ctx.orders。"""
        ...
