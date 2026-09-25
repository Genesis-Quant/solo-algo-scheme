"""control 项目的默认实现。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from backtest import Direction, MarketOrder
from pydantic import Field

from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Order, Target
from scheme.base.projects.control import ControlAlgo
from scheme.base.projects.control import ControlParams as BaseControlParams
from scheme.data.dolphindb.symbols import normalize_assets

__all__ = ["ControlParams", "NoControl"]


class ControlParams(BaseControlParams):
    lot_size: int = Field(default=100, ge=1)


class NoControl[C: ResearchContext[Any]](ControlAlgo[ControlParams, C]):
    def __init__(self, params: ControlParams | None = None) -> None:
        super().__init__(params if params is not None else ControlParams())

    def on_target(self, target: Target) -> None:
        self.ctx.orders = self.target_orders(target)

    def target_orders(self, target: Target) -> list[Order]:
        """以账户权益为分母，将完整目标权重转换为相对当前持仓的订单。"""
        target = normalize_assets(target)
        if any(not np.isfinite(weight) for weight in target.values()):
            raise ValueError("目标权重必须是有限数值")
        quantities: dict[str, int] = {}
        symbols = [s for s, weight in target.items() if weight != 0]
        if symbols:
            prices = self.backtest.get_prices(symbols)
            if any(
                s not in prices or not np.isfinite(prices[s]) or prices[s] <= 0 for s in symbols
            ):
                raise ValueError("目标证券截至当前时点尚无有效价格")
            equity = float(
                self.backtest.session.run(
                    "exec first(totalEquity) from Backtest::getTotalPortfolios(engine)"
                )
            )
            if not np.isfinite(equity) or equity <= 0:
                raise ValueError("账户权益必须为正数，不能据此将目标权重转换为股数")
            lot = self.params.lot_size
            quantities = {
                s: int(np.sign(target[s])) * int(equity * abs(target[s]) / prices[s] // lot) * lot
                for s in symbols
            }
        positions = self.backtest.get_position()
        orders: list[Order] = []
        for symbol in sorted(set(positions) | target.keys()):
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
                    orders.append(
                        MarketOrder(
                            symbol=symbol,
                            time=pd.Timestamp(self.backtest.time).to_pydatetime(),
                            quantity=quantity,
                            direction=direction,
                        )
                    )
        return orders
