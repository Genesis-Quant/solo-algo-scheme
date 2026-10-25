"""在通用回放回调之间调度 Python 研究组件。"""

from collections.abc import Sequence
from math import isfinite
from typing import Any, Generic, TypeVar

from backtest import Backtest as BaseBacktest
from backtest import DosVar
from pydantic import BaseModel
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.base import Algo, Order
from scheme.config import DolphinSettings
from scheme.data.dolphindb.database.session import download, open_session, release
from scheme.data.dolphindb.symbols import engine_symbol, normalize_order

__all__ = ["Backtest"]

C = TypeVar("C", bound=BaseModel, infer_variance=True)


class Backtest(BaseBacktest[C], Generic[C]):
    algos: tuple[Algo[Any, C], ...]

    def __init__(
        self,
        settings: DolphinSettings,
        *,
        engine_name: str,
        config: dict[str, Any],
        ctx: C,
        algos: Sequence[Algo[Any, C]] = (),
    ) -> None:
        self.settings = settings
        self.latest_prices: dict[str, float] = {}
        super().__init__(
            session=open_session(settings),
            engine_name=engine_name,
            config=config,
            ctx=ctx,
            algos=algos,
        )

    def release(self, variables: Sequence[str]) -> None:
        release(self.session, variables)

    def snapshot_fields(self, msg: DosVar) -> dict[str, str]:
        return {"prices": f"(exec dict(string(symbol), lastPrice) from {msg})"}

    def on_advance(self, snapshot: dict[str, Any]) -> None:
        super().on_advance(snapshot)
        for symbol, price in snapshot.get("prices", {}).items():
            if price is not None and isfinite(price) and price > 0:
                self.latest_prices[symbol] = float(price)

    def finish(self, msg: DosVar) -> None:
        try:
            super().finish(msg)
        finally:
            self.release(["soloMessages", "soloDailyMessages"])

    def get_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        return {
            symbol: self.latest_prices[symbol]
            for symbol in map(engine_symbol, symbols)
            if symbol in self.latest_prices
        }

    def get_equity(self) -> float:
        return float(
            download(
                self.session,
                "exec first(totalEquity) from Backtest::getTotalPortfolios(engine)",
                "当前账户权益",
            )
        )

    def _submit_orders(self, orders: Sequence[Order]) -> list[int]:
        return super()._submit_orders([normalize_order(order) for order in orders])

    def cancel_order(
        self, symbol: str = "", orders: Sequence[int] | None = None, label: str = ""
    ) -> None:
        super().cancel_order(engine_symbol(symbol), orders, label)

    def on_event(self, callback: str, payload: Any) -> None:
        try:
            super().on_event(callback, payload)
            while True:
                progressed = False
                for algo in self.algos:
                    if algo.process():
                        progressed = True
                if not progressed:
                    return
        except Exception as error:
            log.error(f"研究事件处理失败：事件={callback}，时间={self.time}；{error_text(error)}")
            raise
