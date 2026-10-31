"""查询、因子分析和回测共用的 DolphinDB 端点及凭据。"""

from query.settings import DolphinSettings as QuerySettings


class DolphinSettings(QuerySettings):
    snapshot_database: str = "dfs://StockSnapshot"
    snapshot_table: str = "snapshot"


__all__ = ["DolphinSettings"]
