"""独立的研究和回测连接配置；环境变量只配置连接端点和凭据。"""

import os

from pydantic import BaseModel, SecretStr


class DolphinSettings(BaseModel):
    host: str = "dolphindb"
    port: int = 8848
    username: str
    password: SecretStr
    database: str = "dfs://CoreData"
    table: str = "coreTable"
    snapshot_database: str = "dfs://StockSnapshot"
    snapshot_table: str = "snapshot"

    @classmethod
    def from_env(cls) -> "DolphinSettings":
        return cls(
            host=os.environ.get("SOLO_DOLPHIN_HOST", "dolphindb"),
            port=os.environ.get("SOLO_DOLPHIN_PORT", 8848),
            username=os.environ["SOLO_DOLPHIN_USERNAME"],
            password=SecretStr(os.environ["SOLO_DOLPHIN_PASSWORD"]),
        )


class BacktestSettings(DolphinSettings):
    host: str = "dolphindb"
    port: int = 8848

    @classmethod
    def from_env(cls) -> "BacktestSettings":
        return cls(
            host=os.environ.get("BACKTEST_DOLPHIN_HOST", "dolphindb"),
            port=os.environ.get("BACKTEST_DOLPHIN_PORT", 8848),
            username=os.environ["BACKTEST_DOLPHIN_USERNAME"],
            password=SecretStr(os.environ["BACKTEST_DOLPHIN_PASSWORD"]),
        )
