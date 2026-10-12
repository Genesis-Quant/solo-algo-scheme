"""Arena 托管股票池：全市场或指数动态池，按两阶段查询逐日判定成分。

指数池先由 codes_query 选出研究区间内的候选并集，第二阶段只查询这些代码，并注入同一个成员节点
``Universe.member``（Arena 的 ``stock_pool_member``）作为逐日成分列，不参与过滤；全市场没有第一阶段，该列恒为 true。
成分来自 CoreData 的指数权重：某日观察到该指数权重时，其他代码缺失的权重先填 0，再按代码前向填充。
"""

from collections.abc import Sequence
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from runtime.apps.query import FactorQuery
from runtime.apps.query import api as query_api
from runtime.apps.query.result import QueryResult
from runtime.database import create_session

from .symbols import engine_symbol

__all__ = ["MEMBERS", "StockPool", "Universe"]

CODES_REFS = {
    "source_ref": "soloUniverseCodesSourceData",
    "computed_ref": "soloUniverseCodesComputedData",
    "filtered_ref": "soloUniverseCodesFilteredData",
    "data_ref": "soloUniverseCodesData",
}


class StockPool(StrEnum):
    """与 Arena 网页股票池选项一一对应。"""

    ALL = "全市场"
    SSE50 = "上证 50"
    CSI300 = "沪深 300"
    CSI500 = "中证 500"
    CSI1000 = "中证 1000"


# 每个股票池的成员节点 DSL，须逐日返回 BOOL，可以写成复合条件；全市场恒为 true，没有第一阶段。
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
    """研究和回测共用的股票池；日期由所属参数提供。

    DSL 用 ``Universe.member`` 引用逐日成分节点，例如 ``"on": Universe.member``。
    """

    model_config = ConfigDict(extra="forbid")

    # 值必须与 Arena 托管节点名一致。
    member: ClassVar[str] = "stock_pool_member"

    pool: StockPool = Field(default=StockPool.ALL, title="股票池")

    def codes_query(self, start: date, end: date) -> FactorQuery | None:
        """第一阶段：结果代码的区间并集作为第二阶段 codes；全市场没有第一阶段。"""
        if self.pool == StockPool.ALL:
            return None
        return FactorQuery.model_validate({
            **_dates(start, end),
            "lookback": "P0D",
            "codes": [],
            "factors": [],
            "derivatives": {self.member: MEMBERS[self.pool]},
            "filters": [self.member],
        })

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
        """第二阶段：注入成员节点，codes 留空，运行时由候选并集或全市场填充。

        成员节点只作为逐日 BOOL 列输出，不参与过滤。下游可把 Universe.member 作为 on，
        放入 filters 只保留当日成分，或读取该列自行处理；DSL 不能重新定义它。
        """
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
        """在同一会话依次执行两阶段查询；结果对象负责关闭会话，需在 with 内读取数据。"""
        dataset = self.dataset_query(
            start, end, lookback=lookback, factors=factors, derivatives=derivatives, filters=filters,
        )
        codes_query = self.codes_query(start, end)
        session = create_session(max_time=query_api.QUERY_SESSION_MAX_TIME)
        try:
            if codes_query is not None:
                codes = query_api.execute_codes_query(codes_query, session=session, **CODES_REFS)
                dataset = dataset.model_copy(update={"codes": codes})
            return query_api.execute_query(dataset, session=session)
        except BaseException:
            session.close()
            raise

    def evaluate(self, start: date, end: date) -> pd.DataFrame:
        """返回 [start, end) 的逐日成分面板：列为候选并集（全市场为查询轴代码），值为当日成员节点。"""
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
