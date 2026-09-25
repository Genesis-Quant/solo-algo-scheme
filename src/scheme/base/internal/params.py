"""所有策略环节共用的研究参数。"""

from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scheme.data.dolphindb.symbols import engine_symbol
from scheme.data.dolphindb.universe import Universe

__all__ = ["StrategyParams", "BacktestParameters"]


def _all_market() -> Universe:
    return Universe(
        derivatives={"member": {"type": "DIRECT", "op": "nullary.true", "fields": {}}},
        filters=["member"],
    )


class StrategyParams(BaseModel):
    """策略统一参数；额外字段保留，供所有 Algo 参数模型读取。"""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（不含）")
    universe: Universe = Field(default_factory=_all_market, title="股票池设置")

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        if self.start >= self.end:
            raise ValueError("日期区间必须满足 start < end，采用 [start, end)")
        self.universe.query(self.start, self.end)
        return self


class BacktestSettings(BaseModel):
    market_data: Literal["stock_daily", "stock_snapshot"] = Field(
        default="stock_daily",
        title="行情类型",
        json_schema_extra={"x-enum-labels": ["日频合成快照", "Tick 快照"]},
    )
    symbols: list[str] | None = Field(default=None, min_length=1, title="行情股票范围")
    benchmark: str | None = Field(default=None, title="基准代码")
    batch_days: int = Field(default=1, ge=1, title="行情加载批次天数")
    config: dict[str, Any] = Field(
        default_factory=lambda: {"cash": 1_000_000.0},
        title="回测引擎配置",
    )

    @model_validator(mode="after")
    def check_config(self) -> "BacktestSettings":
        reserved = {
            "startDate",
            "endDate",
            "dataType",
            "frequency",
            "msgAsTable",
            "msgAsPiecesOnSnapshot",
            "callbackForSnapshot",
            "matchingMode",
            "orderBookMatchingRatio",
            "matchingRatio",
            "benchmark",
        }
        if reserved & self.config.keys():
            raise ValueError(f"config 包含 runtime 管理的字段：{reserved & self.config.keys()}")
        if self.config.get("strategyGroup", "stock") != "stock":
            raise ValueError("当前只支持 stock 引擎")
        if self.symbols is not None:
            self.symbols = [engine_symbol(symbol) for symbol in self.symbols]
            if len(set(self.symbols)) != len(self.symbols):
                raise ValueError("symbols 规范化后不能重复")
        if self.benchmark is not None:
            self.benchmark = engine_symbol(self.benchmark)
        return self


class BacktestParameters(StrategyParams, BacktestSettings):
    """完整回测参数，保留额外字段供各 Algo 读取。"""
