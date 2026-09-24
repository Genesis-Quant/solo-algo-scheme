"""行情源与回测引擎的证券代码转换。"""

__all__ = ["engine_symbol", "source_symbol"]


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
