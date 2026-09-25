"""统一数据查询入口；Tushare 仅在被导入使用时初始化。"""

from typing import TYPE_CHECKING, Any

from .dolphindb.api import backtest, query

if TYPE_CHECKING:
    from .tushare.api import pro as pro
    from .tushare.api import ts_api as ts_api

__all__ = ["query", "backtest", "ts_api", "pro"]


def __getattr__(name: str) -> Any:
    if name in {"ts_api", "pro"}:
        from .tushare.api import pro, ts_api

        globals().update(ts_api=ts_api, pro=pro)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
