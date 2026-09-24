"""研究链路的两阶段事件调度。"""

from collections.abc import Sequence
from functools import cached_property
from math import isfinite
from typing import Any

from backtest import Backtest as BaseBacktest
from backtest import DosVar
from pydantic import BaseModel

from .algo import Algo

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
        """返回已推进行情中各证券最近的有效价；尚无有效行情的证券不返回。"""
        return {
            symbol: self._latest_prices[symbol]
            for symbol in symbols if symbol in self._latest_prices
        }

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
