"""股票池的历史候选证券和逐日成员面板。"""

from datetime import date, timedelta
from enum import StrEnum
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from query.utils import logger as log

from scheme.config import DolphinSettings

from .api import QueryResult, query
from .database.session import download
from .symbols import engine_symbol

__all__ = ["MEMBERS", "StockPool", "Universe"]


class StockPool(StrEnum):
    ALL = "全市场"
    SSE50 = "上证 50"
    CSI300 = "沪深 300"
    CSI500 = "中证 500"
    CSI1000 = "中证 1000"


MEMBERS: dict[StockPool, dict[str, Any]] = {
    StockPool.ALL: {"type": "DIRECT", "op": "nullary.true", "fields": {}, "params": {}},
    StockPool.SSE50: {
        "type": "DIRECT",
        "op": "binary.gt",
        "fields": {"left": "weight_000016SH", "right": 0},
        "params": {},
    },
    StockPool.CSI300: {
        "type": "DIRECT",
        "op": "binary.gt",
        "fields": {"left": "weight_000300SH", "right": 0},
        "params": {},
    },
    StockPool.CSI500: {
        "type": "DIRECT",
        "op": "binary.gt",
        "fields": {"left": "weight_000905SH", "right": 0},
        "params": {},
    },
    StockPool.CSI1000: {
        "type": "DIRECT",
        "op": "binary.gt",
        "fields": {"left": "weight_000852SH", "right": 0},
        "params": {},
    },
}


class Universe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    member: ClassVar[str] = "stock_pool_member"
    pool: StockPool = Field(default=StockPool.ALL, title="股票池")

    def member_query(
        self,
        start: date,
        end: date,
        *,
        session: Any | None = None,
        settings: DolphinSettings | None = None,
    ) -> QueryResult:
        if start >= end:
            raise ValueError("股票池日期必须满足 start < end，采用 [start, end)")
        return query(
            {
                "start_date": start.isoformat(),
                "end_date": (end - timedelta(days=1)).isoformat(),
                "codes": [],
                "derivatives": {self.member: MEMBERS[self.pool]},
                "filters": [self.member],
            },
            session=session,
            settings=settings,
        )

    def codes(
        self,
        start: date,
        end: date,
        *,
        session: Any | None = None,
        settings: DolphinSettings | None = None,
    ) -> list[str]:
        """从已经过滤的成员记录取证券并集，返回 .SH/.SZ 代码。"""
        with self.member_query(start, end, session=session, settings=settings) as result:
            codes = list(
                map(
                    str,
                    download(
                        result.session,
                        f"exec distinct code from {result.data_ref} order by code",
                        f"{self.pool} 历史候选证券",
                    ),
                )
            )
        log.info(f"[股票池] {self.pool} 候选证券已确定：{len(codes)} 只")
        return codes

    def evaluate(
        self,
        start: date,
        end: date,
        *,
        session: Any | None = None,
        settings: DolphinSettings | None = None,
    ) -> pd.DataFrame:
        """返回交易日 × 历史成员面板，非成员日保留 False。"""
        with self.member_query(start, end, session=session, settings=settings) as result:
            days = pd.DatetimeIndex(result.calendar).as_unit("ns")
            members = download(
                result.session,
                f"select date(time) as date, string(code) as code, {self.member} as member from {result.data_ref}",
                f"{self.pool} 逐日成员",
            )
        members["date"] = pd.to_datetime(members["date"])
        members["code"] = members["code"].map(engine_symbol)
        panel = members.pivot(index="date", columns="code", values="member").sort_index(axis=1)
        panel.index = pd.DatetimeIndex(panel.index).as_unit("ns")
        panel = panel.reindex(days).eq(True)
        panel.index.name = "date"
        log.info(
            f"[股票池] {self.pool} 面板已就绪：{len(panel)} 个交易日、{len(panel.columns)} 只历史成员"
        )
        return panel
