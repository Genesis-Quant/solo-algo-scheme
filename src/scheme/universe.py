"""带过滤条件的股票池 DSL，以及按交易日对齐的成员面板。"""

from datetime import date, timedelta

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from runtime.apps.query import Derivative, FactorQuery, execute_query

from .symbols import engine_symbol, source_symbol

__all__ = ["Universe"]


class Universe(BaseModel):
    """日期由 FactorParams / StrategyParams 提供；filters 必须非空。"""

    model_config = ConfigDict(extra="forbid")

    codes: list[str] = Field(default_factory=list)
    lookback: timedelta = timedelta(0)
    factors: list[str] = Field(default_factory=list)
    derivatives: dict[str, SerializeAsAny[Derivative]] = Field(default_factory=dict)
    filters: list[str] = Field(min_length=1)

    def query(self, start: date, end: date) -> FactorQuery:
        """复用 Arena 的 DSL 类型、依赖和过滤条件校验。"""
        return FactorQuery.model_validate(
            {
                **self.model_dump(mode="json"),
                "codes": list(dict.fromkeys(source_symbol(code) for code in self.codes)),
                "start_date": start.isoformat(),
                "end_date": (end - timedelta(days=1)).isoformat(),
            }
        )

    def evaluate(self, start: date, end: date) -> pd.DataFrame:
        """返回 [start, end) 的 bool 面板；列仅包含期间至少一次入池的股票。"""
        with execute_query(self.query(start, end)) as result:
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
