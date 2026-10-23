"""查询、因子分析和回测共用的 DolphinDB 端点及凭据。"""

import os
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator


class DolphinSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    endpoints: tuple[str, ...] = ("dolphindb:8848",)
    username: str
    password: SecretStr
    database: str = "dfs://CoreData"
    table: str = "coreTable"
    snapshot_database: str = "dfs://StockSnapshot"
    snapshot_table: str = "snapshot"

    @field_validator("endpoints", mode="before")
    @classmethod
    def validate_endpoints(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("DolphinDB 至少需要一个端点")
        endpoints = []
        for endpoint in value:
            if not isinstance(endpoint, str):
                raise ValueError("DolphinDB 端点必须是 host:port 字符串")
            endpoint = endpoint.strip()
            if not endpoint or any(char.isspace() or char in "/?#@\\" for char in endpoint):
                raise ValueError("DolphinDB 端点只接受 host:port，IPv6 使用 [host]:port")
            try:
                address = urlsplit(f"//{endpoint}")
                host, port = address.hostname, address.port
            except ValueError:
                raise ValueError("DolphinDB 端点格式无效，IPv6 使用 [host]:port") from None
            if not host or port is None or not 1 <= port <= 65535:
                raise ValueError("DolphinDB 端点必须包含主机和 1 至 65535 的端口")
            if endpoint.rsplit(":", 1)[0].lower() != (f"[{host}]" if ":" in host else host):
                raise ValueError("DolphinDB 端点格式无效，IPv6 使用 [host]:port")
            endpoints.append(f"[{host}]:{port}" if ":" in host else f"{host}:{port}")
        return tuple(dict.fromkeys(endpoints))

    @classmethod
    def from_env(cls) -> "DolphinSettings":
        return cls(
            endpoints=os.environ.get("DOLPHINDB_ENDPOINTS", "dolphindb:8848"),
            username=os.environ["DOLPHINDB_USERNAME"],
            password=SecretStr(os.environ["DOLPHINDB_PASSWORD"]),
        )
