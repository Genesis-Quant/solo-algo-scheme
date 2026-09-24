from typing import Any, Literal

from pydantic import Field, model_validator

from scheme import AlgoComponents, StrategyParams
from scheme.symbols import engine_symbol
from scheme.utils.schema import Package, Task


class BacktestParameters(StrategyParams):
    market_data: Literal["stock_daily", "stock_snapshot"] = "stock_daily"
    symbols: list[str] | None = Field(default=None, min_length=1)
    benchmark: str | None = None
    batch_days: int = Field(default=1, ge=1)
    config: dict[str, Any] = Field(default_factory=lambda: {"cash": 1_000_000.0})

    @model_validator(mode="after")
    def check_config(self) -> "BacktestParameters":
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


class BacktestTask(Task):
    kind: Literal["backtest"] = "backtest"
    algos: AlgoComponents[Package]
    context: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    backtest: BacktestParameters
