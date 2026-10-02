"""策略运行参数及其回测分析参数。"""

from datetime import date
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, model_validator

from scheme.base.internal.context import ResearchContext
from scheme.data.dolphindb.symbols import engine_symbol
from scheme.data.dolphindb.universe import Universe

if TYPE_CHECKING:
    from scheme.config import DolphinSettings
    from scheme.execute.strategy.assembly import Strategy
    from scheme.execute.strategy.result import BacktestResult

__all__ = ["StrategyParams", "StrategyAnalysisParams"]


def _all_market() -> Universe:
    return Universe.model_validate({
        "derivatives": {"member": {"type": "DIRECT", "op": "nullary.true", "fields": {}}},
        "filters": ["member"],
    })


def validate_research(start: date, end: date, universe: Universe) -> None:
    if start >= end:
        raise ValueError("日期区间必须满足 start < end，采用 [start, end)")
    universe.query(start, end)


def validate_backtest(
    config: dict[str, Any], symbols: list[str] | None, benchmark: str | None
) -> tuple[list[str] | None, str | None]:
    reserved = {
        "startDate", "endDate", "dataType", "frequency", "msgAsTable",
        "msgAsPiecesOnSnapshot", "callbackForSnapshot", "matchingMode",
        "matchingRatio", "orderBookMatchingRatio", "benchmark",
    }
    if reserved & config.keys():
        raise ValueError(f"config 包含 runtime 管理的字段：{reserved & config.keys()}")
    if config.get("strategyGroup", "stock") != "stock":
        raise ValueError("当前只支持 stock 引擎")
    if symbols is not None:
        symbols = [engine_symbol(symbol) for symbol in symbols]
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols 规范化后不能重复")
    return symbols, engine_symbol(benchmark) if benchmark is not None else None


class StrategyParams(BaseModel):
    """组装策略所需的统一参数；保留各算法声明的额外字段。"""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（不含）")
    universe: Universe = Field(default_factory=_all_market, title="股票池设置")

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        validate_research(self.start, self.end, self.universe)
        return self


class StrategyAnalysisParams(StrategyParams):
    """策略参数加回测设置；将已组装策略运行成报告。"""

    market_data: Literal["stock_daily", "stock_snapshot"] = Field(
        default="stock_daily", title="行情类型",
        json_schema_extra={"x-enum-labels": ["日频合成快照", "Tick 快照"]},
    )
    symbols: list[str] | None = Field(default=None, min_length=1, title="行情股票范围")
    benchmark: str | None = Field(default=None, title="基准代码")
    batch_days: int = Field(default=1, ge=1, title="行情加载批次天数")
    config: dict[str, Any] = Field(
        default_factory=lambda: {"cash": 1_000_000.0}, title="回测引擎配置"
    )

    @model_validator(mode="after")
    def validate_analysis(self) -> Self:
        self.symbols, self.benchmark = validate_backtest(self.config, self.symbols, self.benchmark)
        return self

    def run[P: StrategyParams, C: ResearchContext[Any]](
        self,
        strategy: "Strategy[P, C]",
        *,
        settings: "DolphinSettings | None" = None,
    ) -> "BacktestResult[Self]":
        from scheme.execute.strategy.api import run_backtest
        from scheme.execute.strategy.assembly import Strategy, parameter_type, project_parameters

        if not isinstance(strategy, Strategy):
            raise TypeError("run 需要已组装的 Strategy")
        checked = set(StrategyParams.model_fields)
        if project_parameters(StrategyParams, self) != project_parameters(StrategyParams, strategy.params):
            raise ValueError("分析参数必须与 Strategy 的运行参数一致")
        for algo in strategy.algos:
            model = parameter_type(type(algo))
            if project_parameters(model, self) != algo.params:
                raise ValueError("分析参数必须与 Strategy 的运行参数一致")
            for name, field in model.model_fields.items():
                checked.add(name)
                alias = field.validation_alias
                aliases = alias.choices if isinstance(alias, AliasChoices) else [alias]
                for value in aliases:
                    if isinstance(value, str):
                        checked.add(value)
                    elif isinstance(value, AliasPath):
                        checked.add(str(value.path[0]))
        analysis_fields = set(StrategyAnalysisParams.model_fields) - set(StrategyParams.model_fields)
        runtime_model = type(strategy.params)
        runtime = project_parameters(runtime_model, self)
        expected = dict(strategy.params)
        actual = {**dict(self), **dict(runtime)}
        for name in runtime_model.model_fields.keys() - checked - analysis_fields:
            if actual[name] != expected[name]:
                raise ValueError("分析参数必须与 Strategy 的运行参数一致")
        for name, field in runtime_model.model_fields.items():
            checked.add(name)
            alias = field.validation_alias
            aliases = alias.choices if isinstance(alias, AliasChoices) else [alias]
            for value in aliases:
                if isinstance(value, str):
                    checked.add(value)
                elif isinstance(value, AliasPath):
                    checked.add(str(value.path[0]))
        for name in (set(expected) | set(self.model_extra or {})) - checked - analysis_fields:
            if name not in expected or name not in actual or actual[name] != expected[name]:
                raise ValueError("分析参数必须与 Strategy 的运行参数一致")
        return run_backtest(strategy.algos, strategy.ctx, self, settings=settings)
