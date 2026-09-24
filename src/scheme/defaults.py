"""默认研究环节：风险平价、接受所有订单、直接下单。"""

from __future__ import annotations

from datetime import time, timedelta
from typing import Any

import numpy as np
import pandas as pd
from backtest import Direction, MarketOrder
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from .algo import ControlAlgo, ExecutionAlgo, OptimizeAlgo
from .context import ResearchContext
from .models import Order, Signal, Target
from .symbols import engine_symbol, normalize_assets, source_symbol

__all__ = [
    "RiskParity",
    "NoControl",
    "DirectExecution",
    "OptimizeParams",
    "ControlParams",
    "risk_parity",
]


def risk_parity(returns: pd.DataFrame) -> tuple[np.ndarray, str | None]:
    count = len(returns.columns)
    if not count:
        return np.array([]), None
    equal = np.full(count, 1 / count)
    valid = returns.dropna()
    if len(valid) < max(20, count + 1):
        return equal, "历史样本不足"
    covariance = valid.cov().to_numpy()
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 1e-12:
        return equal, "协方差退化"
    covariance /= np.trace(covariance)

    def objective(weights: np.ndarray) -> float:
        contribution = weights * (covariance @ weights)
        return float(np.sum((contribution / contribution.sum() - 1 / count) ** 2))

    result = minimize(
        objective,
        equal,
        bounds=[(1e-8, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success or objective(result.x) > 1e-6:
        return equal, "风险平价求解未收敛"
    return result.x, None


class DefaultParams(BaseModel):
    pass


class OptimizeParams(BaseModel):
    lookback_days: int = Field(default=180, ge=30)
    gross_exposure: float = Field(default=0.98, gt=0, le=1)


class ControlParams(BaseModel):
    lot_size: int = Field(default=100, ge=1)


class RiskParity[C: ResearchContext[Any]](OptimizeAlgo[OptimizeParams, C]):
    def __init__(self, params: OptimizeParams | None = None) -> None:
        super().__init__(params or OptimizeParams())

    def on_signal(self, signals: Signal[Any]) -> None:
        self.ctx.target = self.allocate(signals)

    def allocate(self, signals: Signal[Any]) -> Target:
        from runtime.apps.query import execute_query as query

        signals = normalize_assets(signals)
        if any(not isinstance(s, (int, float)) or not np.isfinite(s) for s in signals.values()):
            raise TypeError("默认风险平价接受有符号数值 Signal；其他值类型请提供自定义 Optimize")
        symbols = [s for s, value in signals.items() if value != 0]
        if not symbols:
            return {}
        today = pd.Timestamp(self.backtest.time).date()
        with query(
            {
                "start_date": (today - timedelta(days=self.params.lookback_days)).isoformat(),
                "end_date": (today - timedelta(days=1)).isoformat(),
                "codes": [source_symbol(s) for s in symbols],
                "factors": ["close", "adj_factor"],
            }
        ) as result:
            history = result.data
        history["code"] = history.code.map(engine_symbol)
        if history.empty or not {"close", "adj_factor"}.issubset(history.columns):
            returns = pd.DataFrame(columns=symbols, dtype=float)
        else:
            history["adjusted"] = history.close * history.adj_factor
            prices = history.pivot(index="time", columns="code", values="adjusted").reindex(
                columns=symbols
            )
            returns = prices.pct_change(fill_method=None).mul(
                [np.sign(signals[s]) for s in symbols]
            )
        weights, reason = risk_parity(returns)
        target = {
            s: float(np.sign(signals[s]) * self.params.gross_exposure * w)
            for s, w in zip(symbols, weights, strict=True)
        }
        self.diagnostics.append(
            {
                "time": pd.Timestamp(self.backtest.time),
                "method": "equal_weight" if reason else "risk_parity",
                "reason": reason,
            }
        )
        return target


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


class DirectExecution[C: ResearchContext[Any]](ExecutionAlgo[DefaultParams, C]):
    def __init__(self, params: DefaultParams | None = None) -> None:
        super().__init__(params if params is not None else DefaultParams())

    def process(self) -> bool:
        now = pd.Timestamp(self.backtest.time).time()
        if not (time(9, 30) <= now < time(11, 30) or time(13) <= now < time(15)):
            return False
        return super().process()

    def on_orders(self, orders: list[Order]) -> None:
        now = pd.Timestamp(self.backtest.time).to_pydatetime()
        for pending in orders:
            order = pending.model_copy(update={"time": now})
            self.backtest.submit_order(order)
