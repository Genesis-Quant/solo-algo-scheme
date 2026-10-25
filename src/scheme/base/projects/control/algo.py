"""ControlAlgo 研究项目基类。"""

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from backtest import Direction, MarketOrder
from pydantic import BaseModel

from scheme.base.internal.algo import Algo
from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Order, Target
from scheme.base.projects.execution.algo import ExecutionAlgo
from scheme.data.dolphindb.database.session import download
from scheme.data.dolphindb.symbols import normalize_assets

__all__ = ["ControlAlgo"]

P = TypeVar("P", bound=BaseModel, infer_variance=True)
C = TypeVar("C", bound=ResearchContext[Any], infer_variance=True)


class ControlAlgo(Algo[P, C], Generic[P, C]):
    def process(self) -> bool:
        target = self.ctx.target
        if target is None or self.ctx.orders is not None:
            return False
        if any(
            isinstance(algo, ExecutionAlgo)
            and type(algo).process is ExecutionAlgo.process
            and not algo.can_execute()
            for algo in self.backtest.algos
        ):
            return False
        self.ctx.target = None
        self.on_target(target)
        return True

    @abstractmethod
    def on_target(self, target: Target) -> None:
        """根据目标权重生成并检查订单，将通过的订单写入 ctx.orders。"""
        ...

    def target_orders(self, target: Target, *, lot_size: int = 100) -> list[Order]:
        """以账户权益为分母，将完整目标权重转换为相对当前持仓的订单。"""
        if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
            raise ValueError("每手股数必须是正整数")
        target = normalize_assets(target)
        if any(not np.isfinite(weight) for weight in target.values()):
            raise ValueError("目标权重必须是有限数值")
        quantities: dict[str, int] = {}
        skipped: set[str] = set()
        symbols = [s for s, weight in target.items() if weight != 0]
        if symbols:
            prices = self.backtest.get_prices(symbols)
            skipped = {
                s
                for s in symbols
                if s not in prices or not np.isfinite(prices[s]) or prices[s] <= 0
            }
            if skipped:
                self.diagnostics.append(
                    {
                        "time": pd.Timestamp(self.backtest.time),
                        "method": "missing_price",
                        "reason": f"跳过 {len(skipped)} 只截至当前时点无有效价格的证券，保持现有持仓",
                    }
                )
            symbols = [s for s in symbols if s not in skipped]
        if symbols:
            equity = float(
                download(
                    self.backtest.session,
                    "exec first(totalEquity) from Backtest::getTotalPortfolios(engine)",
                    "当前账户权益",
                )
            )
            if not np.isfinite(equity) or equity <= 0:
                raise ValueError("账户权益必须为正数，不能据此将目标权重转换为股数")
            lot = lot_size
            quantities = {
                s: int(np.sign(target[s])) * int(equity * abs(target[s]) / prices[s] // lot) * lot
                for s in symbols
            }
        positions = self.backtest.get_position()
        closing_orders: list[Order] = []
        opening_orders: list[Order] = []
        for symbol in sorted((set(positions) | target.keys()) - skipped):
            position = positions.get(symbol)
            long = position.longPosition if position else 0
            short = position.shortPosition if position else 0
            quantity = quantities.get(symbol, 0)
            wanted_long, wanted_short = max(quantity, 0), max(-quantity, 0)
            changes = [
                (Direction.SELL_CLOSE, max(long - wanted_long, 0)),
                (Direction.BUY_CLOSE, max(short - wanted_short, 0)),
                (Direction.BUY_OPEN, max(wanted_long - long, 0)),
                (Direction.SELL_OPEN, max(wanted_short - short, 0)),
            ]
            for direction, quantity in changes:
                if quantity:
                    orders = (
                        closing_orders
                        if direction in (Direction.SELL_CLOSE, Direction.BUY_CLOSE)
                        else opening_orders
                    )
                    orders.append(
                        MarketOrder(
                            symbol=symbol,
                            time=pd.Timestamp(self.backtest.time).to_pydatetime(),
                            quantity=quantity,
                            direction=direction,
                        )
                    )
        return closing_orders + opening_orders
