"""Tushare 固定股票列表中的逐日股票池成员，不预筛选区间候选代码。

codes 留空，由 scheme.data.query 复用 Arena Runtime 的 stock_basic 股票列表，
合并上市、退市、暂停上市股票并在进程内缓存；导入模块不联网。
日期轴沿用 Runtime 的真实 close 交易日。成员来自 CoreData 指数权重，
缺失权重按观测日补零后逐代码前向填充；成员列默认仅输出，由调用者选择 filters 或 on。
"""

from collections.abc import Sequence
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from runtime.apps.query import FactorQuery
from runtime.apps.query.result import QueryResult

from .api import query
from .symbols import engine_symbol

__all__ = ["MEMBERS", "StockPool", "Universe"]


class StockPool(StrEnum):
    """研究表单可选的全市场及指数动态池。"""

    ALL = "全市场"
    SSE50 = "上证 50"
    CSI300 = "沪深 300"
    CSI500 = "中证 500"
    CSI1000 = "中证 1000"


MEMBERS: dict[StockPool, dict[str, Any]] = {
    StockPool.ALL: {"type": "DIRECT", "op": "nullary.true", "fields": {}, "params": {}},
    StockPool.SSE50: {
        "type": "DIRECT", "op": "binary.gt",
        "fields": {"left": "weight_000016SH", "right": 0}, "params": {},
    },
    StockPool.CSI300: {
        "type": "DIRECT", "op": "binary.gt",
        "fields": {"left": "weight_000300SH", "right": 0}, "params": {},
    },
    StockPool.CSI500: {
        "type": "DIRECT", "op": "binary.gt",
        "fields": {"left": "weight_000905SH", "right": 0}, "params": {},
    },
    StockPool.CSI1000: {
        "type": "DIRECT", "op": "binary.gt",
        "fields": {"left": "weight_000852SH", "right": 0}, "params": {},
    },
}


def _dates(start: date, end: date) -> dict[str, str]:
    if start >= end:
        raise ValueError("日期区间必须满足 start < end，采用 [start, end)")
    return {"start_date": start.isoformat(), "end_date": (end - timedelta(days=1)).isoformat()}


class Universe(BaseModel):
    """请求 codes 留空，由 query 补齐进程级证券域；member 逐日判定股票池。"""

    model_config = ConfigDict(extra="forbid")
    member: ClassVar[str] = "stock_pool_member"
    pool: StockPool = Field(default=StockPool.ALL, title="股票池")

    def dataset_query(
        self,
        start: date,
        end: date,
        *,
        lookback: timedelta = timedelta(0),
        factors: Sequence[str] = (),
        derivatives: dict[str, Any] | None = None,
        filters: Sequence[str] = (),
    ) -> FactorQuery:
        """注入成员列，不缩小证券域，不隐式添加成员过滤。"""
        derivatives = dict(derivatives or {})
        if self.member in {*factors, *derivatives}:
            raise ValueError(f"{self.member} 由股票池注入，DSL 不能重新定义")
        return FactorQuery.model_validate({
            **_dates(start, end),
            "lookback": lookback,
            "codes": [],
            "factors": list(factors),
            "derivatives": {self.member: MEMBERS[self.pool], **derivatives},
            "filters": list(filters),
        })

    def query(
        self,
        start: date,
        end: date,
        *,
        lookback: timedelta = timedelta(0),
        factors: Sequence[str] = (),
        derivatives: dict[str, Any] | None = None,
        filters: Sequence[str] = (),
    ) -> QueryResult:
        """只执行一次全数据域查询；在结果的 with 内访问数据。"""
        request = self.dataset_query(
            start, end, lookback=lookback, factors=factors, derivatives=derivatives, filters=filters,
        )
        return query(request)

    def evaluate(self, start: date, end: date) -> pd.DataFrame:
        """返回全部查询证券的逐日成员面板，缺行情不删除证券列。"""
        with self.query(start, end) as result:
            members = result.session.run(f"""
                select date(time) as date, string(code) as code, {self.member} as member
                from {result.data_ref}
            """)
        members["date"] = pd.to_datetime(members.date)
        members["code"] = members.code.map(engine_symbol)
        panel = members.pivot(index="date", columns="code", values="member").sort_index(axis=1)
        panel.index = panel.index.as_unit("ns")
        return panel.eq(True)
