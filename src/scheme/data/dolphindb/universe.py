"""带过滤条件的股票池 DSL，以及按交易日对齐的成员面板。"""

from datetime import date, timedelta
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator
from runtime.apps.query import Derivative, FactorQuery

from scheme.data import query

from .symbols import engine_symbol, source_symbol

__all__ = ["StockPool", "Universe"]


class StockPool(StrEnum):
    """Python 可导入的股票池选项；Pydantic 同时将其导出为表单枚举。"""

    CUSTOM = "项目股票池"
    ALL = "全市场"
    SSE50 = "上证 50"
    CSI300 = "沪深 300"
    CSI500 = "中证 500"
    CSI1000 = "中证 1000"


class Universe(BaseModel):
    """日期由 FactorParams / StrategyParams 提供；filters 必须非空。"""

    model_config = ConfigDict(extra="forbid")

    pool: StockPool = Field(default=StockPool.CUSTOM, title="股票池")
    lookback: timedelta = Field(default=timedelta(0), title="回溯周期")
    codes: list[str] = Field(default_factory=list, json_schema_extra={"x-hidden": True})
    factors: list[str] = Field(default_factory=list, json_schema_extra={"x-hidden": True})
    derivatives: dict[str, SerializeAsAny[Derivative]] = Field(
        default_factory=dict, json_schema_extra={"x-hidden": True}
    )
    filters: list[str] = Field(min_length=1, json_schema_extra={"x-hidden": True})

    @model_validator(mode="before")
    @classmethod
    def resolve_pool(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        pool = StockPool(value.get("pool", StockPool.CUSTOM))
        if pool == StockPool.CUSTOM:
            return value
        weights = {
            StockPool.SSE50: "weight_000016SH",
            StockPool.CSI300: "weight_000300SH",
            StockPool.CSI500: "weight_000905SH",
            StockPool.CSI1000: "weight_000852SH",
        }
        member = (
            {"type": "DIRECT", "op": "nullary.true", "fields": {}}
            if pool == StockPool.ALL
            else {
                "type": "DIRECT",
                "op": "binary.gt",
                "fields": {"left": weights[pool], "right": 0},
                "params": {},
            }
        )
        return {
            **value,
            "codes": [],
            "factors": [],
            "derivatives": {"stock_pool_member": member},
            "filters": ["stock_pool_member"],
        }

    def query(self, start: date, end: date) -> FactorQuery:
        """复用 Arena 的 DSL 类型、依赖和过滤条件校验。"""
        return FactorQuery.model_validate(
            {
                **self.model_dump(mode="json", exclude={"pool"}),
                "codes": list(dict.fromkeys(source_symbol(code) for code in self.codes)),
                "start_date": start.isoformat(),
                "end_date": (end - timedelta(days=1)).isoformat(),
            }
        )

    def evaluate(self, start: date, end: date) -> pd.DataFrame:
        """返回 [start, end) 的 bool 面板；列仅包含期间至少一次入池的股票。"""
        with query(self.query(start, end)) as result:
            dates = result.session.run(f"""
                exec distinct date(time) from {result.source_ref}
                where time >= coreOutputStart, time < coreOutputEnd
                order by date(time)
            """)
            members = result.session.run(
                f"select date(time) as date, string(code) as code from {result.data_ref}"
            )
        index = pd.DatetimeIndex(dates, name="date").as_unit("ns")
        if members.empty:
            return pd.DataFrame(index=index, columns=pd.Index([], name="code"), dtype=bool)
        members["date"] = pd.to_datetime(members.date)
        members["code"] = members.code.map(engine_symbol)
        members = members.drop_duplicates(["date", "code"])
        members["member"] = True
        panel = members.pivot(index="date", columns="code", values="member")
        return panel.reindex(index=index, columns=sorted(panel.columns)).notna()
