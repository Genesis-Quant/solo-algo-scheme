"""Tushare SDK 和使用 TUSHARE_TOKEN 初始化的 Pro 客户端。"""

import os

import tushare as ts_api
from tushare.pro.client import DataApi

_token = os.getenv("TUSHARE_TOKEN")
if not _token:
    raise RuntimeError("请先配置 TUSHARE_TOKEN，再导入 Tushare 数据源")

pro: DataApi = ts_api.pro_api(_token)

__all__ = ["pro", "ts_api"]
