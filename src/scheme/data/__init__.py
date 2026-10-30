"""统一数据查询入口；Tushare 仅在被导入使用时初始化。"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .dolphindb.api import query
from .dolphindb.schema import FactorQuery

if TYPE_CHECKING:
    from .tushare.api import pro as pro
    from .tushare.api import ts_api as ts_api

__all__ = ["FactorQuery", "query", "ts_api", "pro"]


def __getattr__(name: str) -> Any:
    if name in {"ts_api", "pro"}:
        module = import_module("scheme.data.tushare.api")
        globals().update(ts_api=module.ts_api, pro=module.pro)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
