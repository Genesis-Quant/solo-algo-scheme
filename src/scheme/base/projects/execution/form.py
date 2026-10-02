"""Execution 项目的回测报告表单。"""

from datetime import date, timedelta
from typing import Literal

from pydantic import Field

from scheme.base.internal.form import ReportForm
from scheme.data.dolphindb.universe import StockPool, Universe

from .params import ExecutionAnalysisParams

__all__ = ["ExecutionReportForm"]


class ExecutionReportForm(ReportForm[ExecutionAnalysisParams]):
    start: date = Field(title="开始日期")
    end: date = Field(title="结束日期（不含）")
    pool: Literal[
        StockPool.ALL, StockPool.SSE50, StockPool.CSI300, StockPool.CSI500, StockPool.CSI1000
    ] = Field(default=StockPool.CSI300, title="股票池")
    lookback: timedelta = Field(default=timedelta(0), title="回溯周期")
    market_data: Literal["stock_daily", "stock_snapshot"] = Field(
        default="stock_daily",
        title="行情类型",
        json_schema_extra={"x-enum-labels": ["日频合成快照", "Tick 快照"]},
    )
    benchmark: str | None = Field(default="000300.SH", title="基准代码")
    batch_days: int = Field(default=1, ge=1, title="行情加载批次天数")
    cash: float = Field(default=1_000_000.0, gt=0, title="初始资金")
    commission: float = Field(default=0.0003, ge=0, title="佣金费率")
    tax: float = Field(default=0.0005, ge=0, title="印花税率")
    model: str = Field(min_length=1, title="策略建模", json_schema_extra={"x-algo-kind": "model"})
    optimize: str = Field(
        min_length=1, title="组合优化", json_schema_extra={"x-algo-kind": "optimize"}
    )
    control: str = Field(
        min_length=1, title="订单风控", json_schema_extra={"x-algo-kind": "control"}
    )

    def build(self) -> ExecutionAnalysisParams:
        return ExecutionAnalysisParams(
            start=self.start,
            end=self.end,
            universe=Universe.model_validate({"pool": self.pool, "lookback": self.lookback}),
            market_data=self.market_data,
            benchmark=self.benchmark,
            batch_days=self.batch_days,
            config={"cash": self.cash, "commission": self.commission, "tax": self.tax},
            model=self.model,
            optimize=self.optimize,
            control=self.control,
        )
