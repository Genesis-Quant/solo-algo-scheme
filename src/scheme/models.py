from dataclasses import dataclass

from backtest import Order

__all__ = ["Asset", "Order", "Signal", "TargetPortfolio"]


type Asset = str
type Signal[T] = dict[Asset, T]


@dataclass(frozen=True)
class TargetPortfolio:
    """完整目标持仓；正数为多头股数，负数为空头股数。"""

    quantities: dict[Asset, int]
