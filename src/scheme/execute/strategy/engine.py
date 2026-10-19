"""研究链路的两阶段事件调度。"""

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
from scheme.data.dolphindb.symbols import engine_symbol, normalize_order

__all__ = ["Backtest"]

C = TypeVar("C", bound=BaseModel, infer_variance=True)


@dataclass
class Progress:
    """按月汇总回测进度；订单、成交和拒单计数在每次月报后清零。"""

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

    @cached_property
    def _latest_prices(self) -> dict[str, float]:
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
            total = len(self.universe.index) if hasattr(self, "parameters") and self.parameters.symbols is None else "未知"
            log.info("回放进度 {}/{} 个交易日，截至 {}；行情时点 {}，订单 {}，成交回报 {}，拒单 {}",
                     progress.completed_days, total, progress.day.date(), progress.moments,
                     progress.total_orders, progress.total_trades, progress.total_rejected)
        if progress.finishing:
            return
        progress.day = day
        progress.moments += 1
        # 新交易日首个行情进入插件后，上一交易日的日终账户记录才会生成。
        self.report_month(moment)
        # 当前行情进入引擎后、订单/成交回报分发前更新；不读取尚未推进的行情。
        prices = self.session.run(f"exec dict(string(symbol), lastPrice) from {msg}")
        for symbol, price in prices.items():
            if price is not None and isfinite(price) and price > 0:
                self._latest_prices[symbol] = float(price)

    def finish(self, msg: DosVar) -> None:
        self.progress.finishing = True
        try:
            super().finish(msg)
        finally:
            self.progress.finishing = False

    def report_month(self, moment: pd.Timestamp) -> None:
        """进入新月份时汇总上月；账户记录截至上月最后一个交易日日终。"""
        progress = self.progress
        month = moment.to_period("M")
        if progress.month is None or month == progress.month:
            progress.month = month
            return
        self.session.upload({"soloProgressDate": np.datetime64(moment.date(), "D")})
        portfolios = self.get_records(
            "select top 1 * from Backtest::getDailyTotalPortfolios(engine) "
            "where tradeDate < soloProgressDate order by tradeDate desc",
            DailyPortfolio,
        )
        if portfolios:
            latest = portfolios[0]
            positions = sum(
                1 for item in self.get_position().values()
                if item.longPosition > 0 or item.shortPosition > 0
            )
            monthly = (
                f"{latest.netValue / progress.net_value - 1:+.2%}"
                if progress.net_value is not None and progress.net_value > 0 else "不可用"
            )
            log.info(
                "{} 完成：净值 {:.4f}（本月 {}），持仓 {} 只，现金 {:.1%}；"
                "订单 {}，成交 {}，拒单 {}",
                progress.month, latest.netValue, monthly,
                positions, latest.cash / latest.totalEquity if latest.totalEquity else 0.0,
                progress.orders, progress.trades, progress.rejected,
            )
            progress.net_value = latest.netValue
        progress.month = month
        progress.orders = progress.trades = progress.rejected = 0

    def get_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        """接受两种后缀，返回以 .XSHE/.XSHG 为键的最近有效价。"""
        return {
            symbol: self._latest_prices[symbol]
            for symbol in map(engine_symbol, symbols)
            if symbol in self._latest_prices
        }

    def submit_order(self, order: Order) -> int:
        """统一证券后缀后下单，不修改传入的订单对象。"""
        order = normalize_order(order)
        try:
            identifier = super().submit_order(order)
        except Exception as error:
            log.error("委托提交失败：时间={}，证券={}，方向={}，数量={}；{}",
                      self.time, order.symbol, order.direction.name, order.quantity, error_text(error))
            raise
        self.progress.orders += 1
        self.progress.total_orders += 1
        log.debug("委托 #{}：时间={}，{} {} {} 股，类型={}",
                  identifier, self.time, order.symbol, order.direction.name,
                  order.quantity, type(order).__name__)
        return identifier

    def cancel_order(
        self,
        symbol: str = "",
        orders: Sequence[int] | None = None,
        label: str = "",
    ) -> None:
        """按证券撤单时接受 .SZ/.SH 和 .XSHE/.XSHG。"""
        super().cancel_order(engine_symbol(symbol), orders, label)

    def on_event(self, callback: str, payload: Any) -> None:
        """逆序通知一次事件，然后正序消费消息直至没有可处理的消息。

        下单生成的回报仍由底层 process_events 读取，当前轮处理完成后才会
        通知下一批回报；此处不追加行情、不递归分发回报。
        """
        if callback == "on_order":
            for report in payload:
                if report.status == OrderStatus.REJECTED:
                    self.progress.rejected += 1
                    self.progress.total_rejected += 1
                    log.warning(
                        "拒单 #{}：时间={}，{} {} {} 股 @ {}；状态={}，插件回报未提供拒单原因",
                        report.orderId, report.updateTime, report.symbol,
                        report.direction.name, report.qty,
                        f"{report.price:.2f}" if report.price else "市价", report.status.name,
                    )
        elif callback == "on_trade":
            self.progress.trades += len(payload)
            self.progress.total_trades += len(payload)
            for trade in payload:
                log.debug("成交 #{}：{}，{} {} {} 股 @ {:.4f}，金额 {:.2f}",
                          trade.orderId, trade.tradeTime, trade.symbol, trade.direction.name,
                          trade.tradeQty, trade.tradePrice, trade.tradeValue)
        try:
            super().on_event(callback, payload)
        except Exception as error:
            log.error("算法回调失败：事件={}，时间={}；{}", callback, self.time, error_text(error))
            raise
        while True:
            progressed = False
            for algo in self.algos:
                try:
                    consumed = algo.process()
                except Exception as error:
                    log.error("研究环节失败：算法={}.{}，时间={}；{}",
                              type(algo).__module__, type(algo).__qualname__, self.time, error_text(error))
                    raise
                if consumed:
                    progressed = True
                    log.debug("研究决策 {}：{}.{} 已处理；待处理 signal={}、target={}、orders={}",
                             self.time, type(algo).__module__, type(algo).__qualname__,
                             len(self.ctx.signal) if self.ctx.signal is not None else 0,
                             len(self.ctx.target) if self.ctx.target is not None else 0,
                             len(self.ctx.orders) if self.ctx.orders is not None else 0)
            if not progressed:
                return
