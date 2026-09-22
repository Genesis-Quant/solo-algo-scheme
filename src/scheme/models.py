from dataclasses import dataclass

from backtest import Direction

__all__ = ["Order", "Recommendation", "TargetPortfolio"]


@dataclass(frozen=True)
class Recommendation:
    """完整推荐集合；分数正负表示多空，空集合表示清仓。"""

    scores: dict[str, float]


@dataclass(frozen=True)
class TargetPortfolio:
    """完整目标持仓；正数为多头股数，负数为空头股数。"""

    quantities: dict[str, int]


@dataclass(frozen=True)
class Order:
    """组合生成的母单，经过风控后交给执行算法。"""

    symbol: str
    quantity: int
    direction: Direction
