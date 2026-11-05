"""DolphinDB连接和研究数据位置；配置类不创建连接。"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")


class DolphinSettings:
    """直接传配置类，运行账号与更新账号分开使用。"""

    endpoints = tuple(
        endpoint.strip() for endpoint in os.getenv(
            "DOLPHINDB_ENDPOINTS",
            f"{os.getenv('DOLPHIN_HOST', 'dolphindb')}:{os.getenv('DOLPHIN_PORT', '8848')}",
        ).split(",") if endpoint.strip()
    )
    username = os.getenv("DOLPHINDB_USERNAME") or os.getenv("DOLPHIN_RUNTIME_USERNAME")
    password = os.getenv("DOLPHINDB_PASSWORD") or os.getenv("DOLPHIN_RUNTIME_PASSWORD")
    worker_username = os.getenv("DOLPHINDB_WORKER_USERNAME") or os.getenv("DOLPHIN_WORKER_USERNAME")
    worker_password = os.getenv("DOLPHINDB_WORKER_PASSWORD") or os.getenv("DOLPHIN_WORKER_PASSWORD")
    core_database = os.getenv("DOLPHIN_CORE_DATABASE", "dfs://CoreData")
    core_table = os.getenv("DOLPHIN_CORE_TABLE", "coreTable")
    barra_database = os.getenv("DOLPHIN_BARRA_DATABASE", "dfs://Barra")
    barra_table = os.getenv("DOLPHIN_BARRA_TABLE", "barra")
    barra_cov_table = os.getenv("DOLPHIN_BARRA_COV_TABLE", "barraCov")
    barra_sp_table = os.getenv("DOLPHIN_BARRA_SP_TABLE", "barraSp")
    snapshot_database = os.getenv("DOLPHIN_SNAPSHOT_DATABASE", "dfs://StockSnapshot")
    snapshot_table = os.getenv("DOLPHIN_SNAPSHOT_TABLE", "snapshot")
