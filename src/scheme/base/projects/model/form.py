"""Model 研究表单：回测设置与默认后续算法。"""

from datetime import date, timedelta
from typing import Literal

from pydantic import Field

from scheme.base.internal.form import ReportForm
from scheme.data.dolphindb.universe import StockPool, Universe

from .params import ModelAnalysisParams

__all__ = ["ModelReportForm"]


class ModelReportForm(ReportForm[ModelAnalysisParams]):
    start: date = Field(default=date(2020, 1, 1), title="开始日期")
    end: date = Field(default=date(2027, 1, 1), title="结束日期（不含）")
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
    optimize: Literal["risk_parity"] = Field(
        default="risk_parity", title="组合优化", json_schema_extra={"x-enum-labels": ["风险平价"]}
    )
    control: Literal["no_control"] = Field(
        default="no_control", title="订单风控", json_schema_extra={"x-enum-labels": ["不风控"]}
    )
    execution: Literal["direct_execution"] = Field(
        default="direct_execution",
        title="算法下单",
        json_schema_extra={"x-enum-labels": ["不拆单"]},
    )

    def build(self) -> ModelAnalysisParams:
        return ModelAnalysisParams(
            start=self.start,
            end=self.end,
            universe=Universe.model_validate({"pool": self.pool, "lookback": self.lookback}),
            market_data=self.market_data,
            benchmark=self.benchmark,
            batch_days=self.batch_days,
            config={"cash": self.cash, "commission": self.commission, "tax": self.tax},
            optimize=self.optimize,
            control=self.control,
            execution=self.execution,
        )
