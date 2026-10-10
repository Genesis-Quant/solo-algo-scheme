"""研究链路的两阶段事件调度。"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from backtest import Backtest as BaseBacktest
from backtest import DailyPortfolio, DosVar, OrderStatus
from pydantic import BaseModel

from scheme.base import Algo, Order
from scheme.data.dolphindb.symbols import engine_symbol, normalize_order
from scheme.log import duration, get_logger, is_enabled, mark_simulation

__all__ = ["Backtest"]

log = get_logger("scheme.backtest")


@dataclass
class Progress:
    """按月汇总回测进度；订单、成交和拒单计数在每次月报后清零。"""

    started: float = field(default_factory=perf_counter)
    month: pd.Period | None = None
    net_value: float | None = 1.0
    orders: int = 0
    trades: int = 0
    rejected: int = 0


class Backtest[C: BaseModel](BaseBacktest[C]):
    algos: tuple[Algo[Any, C], ...]

    @cached_property
    def _latest_prices(self) -> dict[str, float]:
        return {}

    @cached_property
    def progress(self) -> Progress:
        return Progress()

    def advance(self, msg: DosVar) -> None:
        moment = pd.Timestamp(self.time)
        mark_simulation(moment)
        super().advance(msg)
        # 新交易日首个行情进入插件后，上一交易日的日终账户记录才会生成。
        self.report_month(moment)
        # 当前行情进入引擎后、订单/成交回报分发前更新；不读取尚未推进的行情。
        prices = self.session.run(f"exec dict(string(symbol), lastPrice) from {msg}")
        for symbol, price in prices.items():
            if price is not None and isfinite(price) and price > 0:
                self._latest_prices[symbol] = float(price)

    def report_month(self, moment: pd.Timestamp) -> None:
        """进入新月份时汇总上月；账户记录截至上月最后一个交易日日终。"""
        progress = self.progress
        month = moment.to_period("M")
        if progress.month is None or month == progress.month:
            progress.month = month
            return
        if not is_enabled("INFO"):
            progress.month = month
            progress.net_value = None
            progress.orders = progress.trades = progress.rejected = 0
            return
        self.session.upload({"soloProgressDate": np.datetime64(moment.date(), "D")})
        portfolios = self.get_records(
            "select top 1 * from getDailyTotalPortfolios(engine) "
            "where tradeDate < soloProgressDate order by tradeDate desc",
            DailyPortfolio,
        )
        if portfolios:
            latest = portfolios[0]
            positions = sum(
                1 for item in self.get_position().values()
                if item.longPosition > 0 or item.shortPosition > 0
            )
            start = pd.Timestamp(self.config["startDate"])
            end = pd.Timestamp(self.config["endDate"] + np.timedelta64(1, "D"))
            done = min(max((moment - start) / (end - start), 1e-6), 1.0)
            elapsed = perf_counter() - progress.started
            monthly = (
                f"{latest.netValue / progress.net_value - 1:+.2%}"
                if progress.net_value is not None and progress.net_value > 0 else "不可用"
            )
            log.info(
                "{} 完成：净值 {:.4f}（本月 {}），持仓 {} 只，现金 {:.1%}；"
                "订单 {}，成交 {}，拒单 {}；已用 {}，预计剩余 {}",
                progress.month, latest.netValue, monthly,
                positions, latest.cash / latest.totalEquity if latest.totalEquity else 0.0,
                progress.orders, progress.trades, progress.rejected,
                duration(elapsed), duration(elapsed / done * (1 - done)),
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
        self.progress.orders += 1
        return super().submit_order(normalize_order(order))

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
                    log.debug(
                        "拒单 #{} {} {} {} 股 @ {}", report.orderId, report.symbol,
                        report.direction.name, report.qty,
                        f"{report.price:.2f}" if report.price else "市价",
                    )
        elif callback == "on_trade":
            self.progress.trades += len(payload)
        super().on_event(callback, payload)
        while True:
            progressed = False
            for algo in self.algos:
                if algo.process():
                    progressed = True
            if not progressed:
                return
