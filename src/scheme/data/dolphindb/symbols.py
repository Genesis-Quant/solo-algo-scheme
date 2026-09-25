"""行情源与回测引擎的证券代码转换。"""

from collections.abc import Mapping

from backtest import Order

__all__ = ["engine_symbol", "source_symbol", "normalize_assets", "normalize_order"]


def source_symbol(symbol: str) -> str:
    return (
        symbol.removesuffix(".XSHE") + ".SZ"
        if symbol.endswith(".XSHE")
        else (symbol.removesuffix(".XSHG") + ".SH" if symbol.endswith(".XSHG") else symbol)
    )


def engine_symbol(symbol: str) -> str:
    return (
        symbol.removesuffix(".SZ") + ".XSHE"
        if symbol.endswith(".SZ")
        else (symbol.removesuffix(".SH") + ".XSHG" if symbol.endswith(".SH") else symbol)
    )


def normalize_assets[T](assets: Mapping[str, T]) -> dict[str, T]:
    """研究链路统一使用 .XSHE/.XSHG；同一证券的不同别名不得重复出现。"""
    normalized: dict[str, T] = {}
    for symbol, value in assets.items():
        key = engine_symbol(symbol)
        if key in normalized:
            raise ValueError(f"证券代码规范化后重复：{key}")
        normalized[key] = value
    return normalized


def normalize_order(order: Order) -> Order:
    """保留订单子类与全部参数，不修改调用者的订单。"""
    symbol = engine_symbol(order.symbol)
    return order if symbol == order.symbol else order.model_copy(update={"symbol": symbol})
