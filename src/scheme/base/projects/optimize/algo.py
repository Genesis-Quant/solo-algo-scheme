"""OptimizeAlgo 研究项目基类。"""

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from scheme.base.internal.algo import Algo
from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Signal
from scheme.data.dolphindb.symbols import normalize_assets

__all__ = ["OptimizeAlgo"]

P = TypeVar("P", bound=BaseModel, infer_variance=True)
C = TypeVar("C", bound=ResearchContext[Any], infer_variance=True)


class OptimizeAlgo(Algo[P, C], Generic[P, C]):
    def process(self) -> bool:
        signals = self.ctx.signal
        if signals is None or self.ctx.target is not None:
            return False
        signals = normalize_assets(signals)
        self.ctx.signal = None
        self.on_signal(signals)
        return True

    @abstractmethod
    def on_signal(self, signals: Signal[Any]) -> None:
        """根据已消费的 signal 计算目标资产权重，写入 ctx.target。"""
        ...
