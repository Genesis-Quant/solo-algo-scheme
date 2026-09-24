"""沿用 Arena 的 DolphinDB 环境变量命名，不包含数据采集配置。"""

import os

from pydantic import BaseModel, SecretStr


class DolphinSettings(BaseModel):
    host: str
    port: int = 8848
    username: str
    password: SecretStr
    database: str = "dfs://CoreData"
    table: str = "coreData"
    snapshot_database: str = "dfs://StockSnapshot"
    snapshot_table: str = "snapshot"

    @classmethod
    def from_env(cls) -> "DolphinSettings":
        return cls(
            host=os.environ["DOLPHIN_HOST"],
            port=int(os.getenv("DOLPHIN_PORT", "8848")),
            username=os.environ["DOLPHIN_RUNTIME_USERNAME"],
            password=SecretStr(os.environ["DOLPHIN_RUNTIME_PASSWORD"]),
            database=os.getenv("DOLPHIN_CORE_DATABASE", "dfs://CoreData"),
            table=os.getenv("DOLPHIN_CORE_TABLE", "coreData"),
            snapshot_database=os.getenv("DOLPHIN_SNAPSHOT_DATABASE", "dfs://StockSnapshot"),
            snapshot_table=os.getenv("DOLPHIN_SNAPSHOT_TABLE", "snapshot"),
        )
