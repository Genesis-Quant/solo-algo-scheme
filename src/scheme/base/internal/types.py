from typing import TypeAlias, TypeVar

from backtest import Order

__all__ = ["Asset", "Order", "Signal", "Target"]

T = TypeVar("T")

Asset: TypeAlias = str
Signal: TypeAlias = dict[Asset, T]
Target: TypeAlias = dict[Asset, float]
