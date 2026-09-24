"""直接导出 Arena 查询和回测接口，结果支持 with。"""

from runtime.apps.backtest import run_backtest as backtest
from runtime.apps.query import execute_query as query

__all__ = ["query", "backtest"]
