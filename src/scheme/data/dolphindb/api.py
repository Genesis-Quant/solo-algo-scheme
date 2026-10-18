"""导出 Arena 查询与会话内联回测接口，结果支持 with。"""

from runtime.apps.query import execute_query as query

from .backtest import backtest

__all__ = ["query", "backtest"]
