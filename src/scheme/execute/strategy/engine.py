"""研究事件调度、回测进度及统一数据交互。"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from backtest import Backtest as BaseBacktest
from backtest import DailyPortfolio, DosVar, OrderStatus
from pydantic import BaseModel
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.base import Algo, Order
from scheme.config import DolphinSettings
from scheme.data.dolphindb.database.session import download, open_session, release, upload
from scheme.data.dolphindb.symbols import engine_symbol, normalize_order

__all__ = ["Backtest"]

C = TypeVar("C", bound=BaseModel, infer_variance=True)


@dataclass
class Progress:
    month: pd.Period | None = None
    net_value: float | None = 1.0
    orders: int = 0
    trades: int = 0
    rejected: int = 0
    day: pd.Timestamp | None = None
    completed_days: int = 0
    moments: int = 0
    total_orders: int = 0
    total_trades: int = 0
    total_rejected: int = 0
    finishing: bool = False


class Backtest(BaseBacktest[C], Generic[C]):
    algos: tuple[Algo[Any, C], ...]

    def __init__(
        self,
        settings: DolphinSettings,
        *,
        engine_name: str,
        config: dict[str, Any],
        ctx: C,
        algos: Sequence[Algo[Any, C]] = (),
    ) -> None:
        self.settings = settings
        super().__init__(
            session=open_session(settings),
            engine_name=engine_name,
            config=config,
            ctx=ctx,
            algos=algos,
        )

    def upload(self, variables: dict[str, Any], label: str) -> None:
        upload(self.session, variables, label)

    def download(self, expression: str, label: str) -> Any:
        return download(self.session, expression, label)

    def release(self, variables: Sequence[str]) -> None:
        release(self.session, variables)

    @cached_property
    def latest_prices(self) -> dict[str, float]:
        return {}

    @cached_property
    def progress(self) -> Progress:
        return Progress()

    def advance(self, msg: DosVar) -> None:
        moment = pd.Timestamp(self.time)
        super().advance(msg)
        progress = self.progress
        day = moment.normalize()
        if progress.day is not None and (day != progress.day or progress.finishing):
            progress.completed_days += 1
            log.info(
                f"[回放] 已完成 {progress.completed_days} 个交易日，截至 {progress.day.date()}；"
                f"行情时点 {progress.moments}，订单 {progress.total_orders}，"
                f"成交回报 {progress.total_trades}，拒单 {progress.total_rejected}"
            )
        if progress.finishing:
            return
        progress.day = day
        progress.moments += 1
        self.report_month(moment)
        prices = self.download(f"exec dict(string(symbol), lastPrice) from {msg}", "当前行情价格")
        for symbol, price in prices.items():
            if price is not None and isfinite(price) and price > 0:
                self.latest_prices[symbol] = float(price)

    def finish(self, msg: DosVar) -> None:
        self.progress.finishing = True
        try:
            super().finish(msg)
        finally:
            self.progress.finishing = False
            self.release(["soloMessages"])

    def report_month(self, moment: pd.Timestamp) -> None:
        progress = self.progress
        month = moment.to_period("M")
        if progress.month is None or month == progress.month:
            progress.month = month
            return
        try:
            self.upload({"progressDate": np.datetime64(moment.date(), "D")}, "回测月报截止日")
            portfolios = self.get_records(
                "select top 1 * from Backtest::getDailyTotalPortfolios(engine) "
                "where tradeDate < progressDate order by tradeDate desc",
                DailyPortfolio,
            )
        finally:
            self.release(["progressDate"])
        if portfolios:
            latest = portfolios[0]
            positions = sum(
                item.longPosition > 0 or item.shortPosition > 0
                for item in self.get_position().values()
            )
            monthly = (
                f"{latest.netValue / progress.net_value - 1:+.2%}"
                if progress.net_value is not None and progress.net_value > 0
                else "不可用"
            )
            cash = latest.cash / latest.totalEquity if latest.totalEquity else 0.0
            log.info(
                f"[月报] {progress.month}：净值 {latest.netValue:.4f}（本月 {monthly}），"
                f"持仓 {positions} 只，现金 {cash:.1%}；"
                f"订单 {progress.orders}，成交 {progress.trades}，拒单 {progress.rejected}"
            )
            progress.net_value = latest.netValue
        progress.month = month
        progress.orders = progress.trades = progress.rejected = 0

    def get_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        return {
            symbol: self.latest_prices[symbol]
            for symbol in map(engine_symbol, symbols)
            if symbol in self.latest_prices
        }

    def submit_order(self, order: Order) -> int:
        identifier = super().submit_order(normalize_order(order))
        self.progress.orders += 1
        self.progress.total_orders += 1
        return identifier

    def cancel_order(
        self, symbol: str = "", orders: Sequence[int] | None = None, label: str = ""
    ) -> None:
        super().cancel_order(engine_symbol(symbol), orders, label)

    def on_event(self, callback: str, payload: Any) -> None:
        if callback == "on_order":
            for report in payload:
                if report.status == OrderStatus.REJECTED:
                    self.progress.rejected += 1
                    self.progress.total_rejected += 1
                    price = f"{report.price:.2f}" if report.price else "市价"
                    log.warning(
                        f"拒单 #{report.orderId}：{report.updateTime}，{report.symbol} "
                        f"{report.direction.name} {report.qty} 股 @ {price}；插件回报未提供拒单原因"
                    )
        elif callback == "on_trade":
            self.progress.trades += len(payload)
            self.progress.total_trades += len(payload)
        try:
            super().on_event(callback, payload)
            while True:
                progressed = False
                for algo in self.algos:
                    if algo.process():
                        progressed = True
                if not progressed:
                    return
        except Exception as error:
            log.error(f"研究事件处理失败：事件={callback}，时间={self.time}；{error_text(error)}")
            raise
