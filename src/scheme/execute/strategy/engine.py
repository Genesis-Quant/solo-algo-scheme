"""研究事件调度和当前行情、账户快照。"""

from collections.abc import Sequence
from math import isfinite
from typing import Any, Generic, TypeVar

from backtest import Backtest as BaseBacktest
from backtest import DosVar, Position
from pydantic import BaseModel
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.base import Algo, Order
from scheme.config import DolphinSettings
from scheme.data.dolphindb.database.session import download, open_session, release
from scheme.data.dolphindb.symbols import engine_symbol, normalize_order

__all__ = ["Backtest"]

C = TypeVar("C", bound=BaseModel, infer_variance=True)


class EngineSession:
    """直接执行 DOS 可能改变账户和事件队列，快照只能在下一次响应后重新确认。"""

    def __init__(self, session: Any, owner: Any) -> None:
        self.raw = session
        self.owner = owner

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.owner.flush_pending_events(session=self.raw)
        self.owner.invalidate_snapshot()
        return self.raw.run(*args, **kwargs)

    def upload(self, *args: Any, **kwargs: Any) -> Any:
        self.owner.flush_pending_events(session=self.raw)
        self.owner.invalidate_snapshot()
        return self.raw.upload(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.raw, name)
        if not callable(value):
            return value

        def invoke(*args: Any, **kwargs: Any) -> Any:
            self.owner.flush_pending_events(session=self.raw)
            self.owner.invalidate_snapshot()
            return value(*args, **kwargs)

        return invoke


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
        self._account_snapshot: dict[str, Any] | None = None
        super().__init__(
            session=open_session(settings),
            engine_name=engine_name,
            config=config,
            ctx=ctx,
            algos=algos,
        )
        self.session = EngineSession(self.session, self)
        self.trusted_event_cache = True
        self.deferred_trade_ack = True

    def invalidate_snapshot(self) -> None:
        self._account_snapshot = None
        self.invalidate_events()

    def release(self, variables: Sequence[str]) -> None:
        release(self.session, variables)

    def snapshot_fields(self, msg: DosVar) -> dict[str, str]:
        return {
            "prices": f"(exec dict(string(symbol), lastPrice) from {msg})",
            "equity": "(exec first(totalEquity) from Backtest::getTotalPortfolios(engine))",
            "positions": "(def(records) { return dict(records.columnNames(), records.values()) })"
            "(Backtest::getPosition(engine))",
        }

    def on_advance(self, snapshot: dict[str, Any]) -> None:
        super().on_advance(snapshot)
        self._account_snapshot = snapshot
        for symbol, price in snapshot.get("prices", {}).items():
            if price is not None and isfinite(price) and price > 0:
                self.latest_prices[symbol] = float(price)

    def finish(self, msg: DosVar) -> None:
        try:
            super().finish(msg)
        finally:
            self.release(["soloMessages"])

    def get_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        return {
            symbol: self.latest_prices[symbol]
            for symbol in map(engine_symbol, symbols)
            if symbol in self.latest_prices
        }

    def get_equity(self) -> float:
        if self._account_snapshot is not None:
            return float(self._account_snapshot["equity"])
        return float(
            download(
                self.session,
                "exec first(totalEquity) from Backtest::getTotalPortfolios(engine)",
                "当前账户权益",
            )
        )

    def get_position(self) -> dict[str, Position]:
        if self._account_snapshot is None:
            return super().get_position()
        columns = self._account_snapshot["positions"]
        return {
            position.symbol: position
            for row in zip(*columns.values(), strict=True)
            for position in [Position.model_validate(dict(zip(columns, row, strict=True)))]
        }

    def submit_orders(self, orders: Sequence[Order]) -> list[int]:
        self._account_snapshot = None
        return super().submit_orders([normalize_order(order) for order in orders])

    def cancel_order(
        self, symbol: str = "", orders: Sequence[int] | None = None, label: str = ""
    ) -> None:
        self._account_snapshot = None
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
