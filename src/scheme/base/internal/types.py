from backtest import Order

__all__ = ["Asset", "Order", "Signal", "Target"]

type Asset = str
type Signal[T] = dict[Asset, T]
type Target = dict[Asset, float]
