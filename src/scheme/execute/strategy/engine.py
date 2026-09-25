"""研究链路的两阶段事件调度。"""

from collections.abc import Sequence
from functools import cached_property
from math import isfinite
from typing import Any

from backtest import Backtest as BaseBacktest
from backtest import DosVar
from pydantic import BaseModel

from scheme.base import Algo, Order
from scheme.data.dolphindb.symbols import engine_symbol, normalize_order

__all__ = ["Backtest"]


class Backtest[C: BaseModel](BaseBacktest[C]):
    algos: tuple[Algo[Any, C], ...]

    @cached_property
    def _latest_prices(self) -> dict[str, float]:
        return {}

    def advance(self, msg: DosVar) -> None:
        super().advance(msg)
        # 当前行情进入引擎后、订单/成交回报分发前更新；不读取尚未推进的行情。
        prices = self.session.run(f"exec dict(string(symbol), lastPrice) from {msg}")
        for symbol, price in prices.items():
            if price is not None and isfinite(price) and price > 0:
                self._latest_prices[symbol] = float(price)

    def get_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        """接受两种后缀，返回以 .XSHE/.XSHG 为键的最近有效价。"""
        return {
            symbol: self._latest_prices[symbol]
            for symbol in map(engine_symbol, symbols)
            if symbol in self._latest_prices
        }

    def submit_order(self, order: Order) -> int:
        """统一证券后缀后下单，不修改传入的订单对象。"""
        return super().submit_order(normalize_order(order))

    def cancel_order(
        self,
        symbol: str = "",
        orders: Sequence[int] | None = None,
        label: str = "",
    ) -> None:
        """按证券撤单时接受 .SZ/.SH 和 .XSHE/.XSHG。"""
        super().cancel_order(engine_symbol(symbol), orders, label)

    def on_event(self, callback: str, payload: Any) -> None:
        """逆序通知一次事件，然后正序消费消息直至没有可处理的消息。

        下单生成的回报仍由底层 process_events 读取，当前轮处理完成后才会
        通知下一批回报；此处不追加行情、不递归分发回报。
        """
        super().on_event(callback, payload)
        while True:
            progressed = False
            for algo in self.algos:
                if algo.process():
                    progressed = True
            if not progressed:
                return
