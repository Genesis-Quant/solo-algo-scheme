"""所有研究 Algo 的公共基类。"""

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from backtest import Algo as BaseAlgo
from pydantic import BaseModel

if TYPE_CHECKING:
    from scheme.execute.strategy.engine import Backtest

__all__ = ["Algo"]

P = TypeVar("P", infer_variance=True)
C = TypeVar("C", bound=BaseModel, infer_variance=True)


class Algo(BaseAlgo[P, C], Generic[P, C]):
    backtest: "Backtest[C]"

    def __init__(self, params: P) -> None:
        super().__init__(params)
        self.diagnostics: list[dict[str, Any]] = []

    def process(self) -> bool:
        """处理一条 Context 消息；处理过返回 True，没有可处理的消息返回 False。"""
        return False
