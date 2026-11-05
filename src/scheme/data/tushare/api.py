"""Tushare SDK和使用TUSHARE_TOKEN的Pro客户端；导入时不请求数据。"""

import os

import tushare
from tushare.pro.client import DataApi

pro = DataApi(token=os.getenv("TUSHARE_TOKEN") or "")

__all__ = ["pro", "tushare"]
