"""研究环节接口：事件通知逆序，Context 消息消费正序。"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from backtest import Algo as BaseAlgo
from pydantic import BaseModel

from .context import ResearchContext
from .models import Order, Signal, Target
from .symbols import normalize_assets, normalize_order

if TYPE_CHECKING:
    from .backtest import Backtest

__all__ = ["Algo", "ModelAlgo", "OptimizeAlgo", "ControlAlgo", "ExecutionAlgo"]


class Algo[P, C: BaseModel](BaseAlgo[P, C]):
    backtest: "Backtest[C]"

    def __init__(self, params: P) -> None:
        super().__init__(params)
        self.diagnostics: list[dict[str, Any]] = []

    def process(self) -> bool:
        """处理一条 Context 消息；处理过返回 True，没有可处理的消息返回 False。"""
        return False


class ModelAlgo[P: BaseModel, C: ResearchContext[Any]](Algo[P, C]):
    """在事件回调内更新 signal；其他环节同样可以更新 signal。"""


class OptimizeAlgo[P: BaseModel, C: ResearchContext[Any]](Algo[P, C]):
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
