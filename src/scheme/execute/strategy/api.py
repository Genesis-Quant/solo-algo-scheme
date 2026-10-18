from collections.abc import Sequence
from datetime import timedelta
from functools import cached_property
from time import perf_counter
from typing import Any, Generic, TypeVar
from uuid import uuid4

import numpy as np
import pandas as pd
from backtest import (
    DailyPortfolio,
    DailyPosition,
    DailyTradingStatistics,
    Direction,
    DosVar,
    OrderStatus,
    TradeDetail,
)
from pydantic import BaseModel

from scheme.base import Algo, ResearchContext, StrategyAnalysisParams
from scheme.config import DolphinSettings
from scheme.data import query
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol
from scheme.execute.strategy.assembly import validate_context
from scheme.execute.strategy.engine import Backtest
from scheme.log import duration, get_logger, mark_simulation

from .result import BacktestResult

log = get_logger("scheme.backtest")

BacktestC = TypeVar("BacktestC", bound=BaseModel, infer_variance=True)
C = TypeVar("C", bound=ResearchContext[Any])
A = TypeVar("A", bound=StrategyAnalysisParams)


def _costs(config: dict[str, Any]) -> dict[str, float]:
    defaults = {"cash": 1_000_000.0, "commission": 0.0003, "tax": 0.0005}
    return {key: float(config.get(key, value)) for key, value in defaults.items()}


class ResearchBacktest(Backtest[BacktestC], Generic[BacktestC]):
    def __init__(
        self,
        settings: DolphinSettings,
        parameters: StrategyAnalysisParams,
        ctx: BacktestC,
        algos: Sequence[Algo[Any, BacktestC]],
    ) -> None:
        self.settings = settings
        self.parameters = parameters
        config = {
            "strategyGroup": "stock",
            "latency": np.int32(0),
            "enableSellCloseRestrict": True,
            **parameters.config,
            **_costs(parameters.config),
            "startDate": parameters.start,
            "endDate": parameters.end - timedelta(days=1),
            "dataType": np.int32(1),
            "frequency": np.int32(0),
            "msgAsTable": True,
            "msgAsPiecesOnSnapshot": True,
            "callbackForSnapshot": np.int32(0),
            "matchingMode": np.int32(1),
            "matchingRatio": 0.0,
            "orderBookMatchingRatio": 1.0,
        }
        for key, value in config.items():
            if type(value) is int and key not in {"cash", "commission", "tax"}:
                config[key] = np.int32(value)
        super().__init__(
            settings.host,
            settings.port,
            settings.username,
            settings.password.get_secret_value(),
            engine_name="solo_" + uuid4().hex,
            config=config,
            ctx=ctx,
            algos=algos,
        )
        try:
            self.session.run("""
                // Daily open/close synthetic snapshots. Price limits must be supplied by source data.
                def soloDailyMessages(daily) {
                    openRows = select code, temporalAdd(temporalAdd(timestamp(time), 9, "h"), 30, "m") as timestamp,
                        open as lastPrice, pre_close as prevClosePrice,
                        up_limit as upLimitPrice, down_limit as downLimitPrice from daily
                    closeRows = select code, temporalAdd(timestamp(time), 15, "h") as timestamp,
                        close as lastPrice, pre_close as prevClosePrice,
                        up_limit as upLimitPrice, down_limit as downLimitPrice from daily
                    rows = unionAll(openRows, closeRows).sortBy!(`timestamp`code)
                    quantities = take(1000000000l, rows.rows())
                    ends = int(1..rows.rows())
                    return table(string(rows.code) as symbol,
                        iif(endsWith(string(rows.code), ".XSHE"), "XSHE", "XSHG") as symbolSource,
                        rows.timestamp, double(rows.lastPrice) as lastPrice,
                        double(rows.prevClosePrice) as prevClosePrice,
                        double(rows.upLimitPrice) as upLimitPrice,
                        double(rows.downLimitPrice) as downLimitPrice,
                        quantities as totalBidQty, quantities as totalOfferQty,
                        arrayVector(ends, double(rows.lastPrice)) as bidPrice,
                        arrayVector(ends, double(rows.lastPrice)) as offerPrice,
                        arrayVector(ends, quantities) as bidQty,
                        arrayVector(ends, quantities) as offerQty)
                }

                def soloTickMessages(raw, reference, universe) {
                    if (raw.rows() == 0) return table(array(TIMESTAMP,0) as timestamp)
                    quotes = table(string(raw.SecurityID) + iif(string(raw.Market)=="SZ", ".XSHE", ".XSHG") as symbol,
                        iif(string(raw.Market)=="SZ", "XSHE", "XSHG") as symbolSource,
                        raw.TradeDate as tradeDate, concatDateTime(raw.TradeDate, raw.TradeTime) as timestamp,
                        double(raw.LastPrice) as lastPrice,
                        fixedLengthArrayVector(double(raw.BidPrice1),double(raw.BidPrice2),double(raw.BidPrice3),double(raw.BidPrice4),double(raw.BidPrice5)) as bidPrice,
                        fixedLengthArrayVector(double(raw.AskPrice1),double(raw.AskPrice2),double(raw.AskPrice3),double(raw.AskPrice4),double(raw.AskPrice5)) as offerPrice,
                        fixedLengthArrayVector(long(raw.BidVolume1),long(raw.BidVolume2),long(raw.BidVolume3),long(raw.BidVolume4),long(raw.BidVolume5)) as bidQty,
                        fixedLengthArrayVector(long(raw.AskVolume1),long(raw.AskVolume2),long(raw.AskVolume3),long(raw.AskVolume4),long(raw.AskVolume5)) as offerQty,
                        long(raw.BidVolume1+raw.BidVolume2+raw.BidVolume3+raw.BidVolume4+raw.BidVolume5) as totalBidQty,
                        long(raw.AskVolume1+raw.AskVolume2+raw.AskVolume3+raw.AskVolume4+raw.AskVolume5) as totalOfferQty)
                    quotes = select * from quotes where symbol in universe
                    refs = select date(time) as tradeDate, string(code) as symbol, double(pre_close) as prevClosePrice,
                        double(up_limit) as upLimitPrice, double(down_limit) as downLimitPrice from reference
                    quotes = lj(quotes, refs, `tradeDate`symbol)
                    if (any(isNull(quotes.prevClosePrice) || isNull(quotes.upLimitPrice) || isNull(quotes.downLimitPrice)))
                        throw "Snapshot reference prices are missing"
                    return select symbol, symbolSource, timestamp, lastPrice, prevClosePrice,
                        upLimitPrice, downLimitPrice, totalBidQty, totalOfferQty,
                        bidPrice, offerPrice, bidQty, offerQty from quotes order by timestamp, symbol
                }
            """)
        except Exception:
            self.close()
            raise

    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        codes = self.parameters.symbols
        if codes is None:
            codes = self.universe.columns.tolist()
        symbols = [engine_symbol(s) for s in codes]
        if not symbols:
            self.session.run("soloMessages = table(array(TIMESTAMP,0) as timestamp)")
            return DosVar("soloMessages")
        fields = ["pre_close", "up_limit", "down_limit"]
        if self.parameters.market_data != "stock_snapshot":
            fields = ["open", "close", *fields]
        with query(
            {
                "start_date": str(start.astype("datetime64[D]")),
                "end_date": str(end.astype("datetime64[D]") - np.timedelta64(1, "D")),
                # 非交易日批次也能取得日历，最终投影仍只返回本批次。
                "lookback": "P30D",
                "codes": [source_symbol(s) for s in symbols],
                "factors": fields,
            }
        ) as result:
            frame = result.data
        frame["code"] = frame.code.map(engine_symbol)
        if self.parameters.market_data == "stock_snapshot":
            self.session.upload({"soloSnapshotReference": frame})
            self.session.upload(
                {
                    "soloSnapshotDB": self.settings.snapshot_database,
                    "soloSnapshotTable": self.settings.snapshot_table,
                    "soloBatchStart": start,
                    "soloBatchEnd": end,
                    "soloUniverse": symbols,
                    "soloSecurityIDs": [s.split(".")[0] for s in symbols],
                }
            )
            self.session.run("""
                soloSnapshots = loadTable(soloSnapshotDB, soloSnapshotTable)
                soloSnapshotRows = select * from soloSnapshots
                    where TradeDate >= date(soloBatchStart), TradeDate < date(soloBatchEnd),
                          SecurityID in soloSecurityIDs, LastPrice > 0,
                          (TradeTime between 09:30:00.000:11:30:00.000 or
                           TradeTime between 13:00:00.000:15:00:00.000)
                soloMessages = soloTickMessages(soloSnapshotRows, soloSnapshotReference, soloUniverse)
            """)
        else:
            columns = ["open", "close", "pre_close", "up_limit", "down_limit"]
            valid = (frame[columns].notna() & (frame[columns] > 0)).all(axis=1)
            skipped = int((~valid).sum())
            if skipped:
                log.warning("日线价格缺失或非正，{} 条证券日记录不合成快照，当日不可交易", skipped)
            frame = frame[valid]
            if frame.empty:
                self.session.run("soloMessages = table(array(TIMESTAMP,0) as timestamp)")
                return DosVar("soloMessages")
            self.session.upload({"soloDaily": frame})
            self.session.run("soloMessages = soloDailyMessages(soloDaily)")
        return DosVar("soloMessages")

    @cached_property
    def universe(self) -> pd.DataFrame:
        """供 Algo 按日期读取动态股票池；行情覆盖期间成员的并集。"""
        started = perf_counter()
        members = self.parameters.universe.evaluate(self.parameters.start, self.parameters.end)
        log.info(
            "股票池 {}：{} 个交易日，期间成员 {} 只，用时 {:.1f} 秒",
            self.parameters.universe.pool, len(members.index), len(members.columns),
            perf_counter() - started,
        )
        return members


def records_frame(records: Sequence[BaseModel], model: type[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump() for row in records], columns=list(model.model_fields))


def run_backtest(
    algos: Sequence[Algo[Any, C]],
    ctx: C,
    parameters: A,
    *,
    settings: DolphinSettings | None = None,
    engine_type: type[ResearchBacktest[C]] = ResearchBacktest,
) -> BacktestResult[A]:
    """逆序事件回调后正序消费 Context 消息；仍允许 Algo 直接下单。"""
    settings = settings or DolphinSettings.from_env()
    if not algos:
        raise ValueError("至少需要一个 Algo")
    for algo in algos:
        if not isinstance(algo, Algo):
            raise TypeError("算法必须继承当前 scheme 的 Algo")
        validate_context(algo, ctx)
    config = _costs(parameters.config)
    log.info(
        "回测开始：{} 至 {}（不含），{}，每批 {} 天；资金 {:,.0f}，佣金 {:.4%}，印花税 {:.4%}，基准 {}；"
        "算法 {}",
        parameters.start, parameters.end, parameters.market_data, parameters.batch_days,
        config["cash"], config["commission"], config["tax"], parameters.benchmark or "无",
        " → ".join(type(algo).__name__ for algo in algos),
    )
    started = perf_counter()
    try:
        result = _run(algos, ctx, parameters, settings, engine_type)
    finally:
        mark_simulation(None)
    try:
        _summarize(result, perf_counter() - started)
    except Exception:
        # 汇总日志不能让已经完成的回测失败。
        log.opt(exception=True).warning("回测结果汇总失败")
    return result


def _summarize(result: BacktestResult[Any], seconds: float) -> None:
    """回测结束的收益、费用和订单汇总；拒单按插件返回的原因归类为警告。"""
    portfolios = result.daily_portfolios
    if portfolios.empty:
        log.warning("回测没有生成账户记录，用时 {}", duration(seconds))
        return
    net_value = portfolios.netValue
    drawdown = (1 - net_value / net_value.cummax().clip(lower=1.0)).max()
    benchmark = (
        f"{portfolios.benchmarkNetValue.iloc[-1] - 1:+.2%}"
        if "benchmarkNetValue" in portfolios else "无"
    )
    details = result.trade_details
    orders = details.orderId.nunique()
    filled = details.loc[details.orderStatus == OrderStatus.FILLED, "orderId"].nunique()
    rejected = details[details.orderStatus == OrderStatus.REJECTED]
    log.info(
        "回测完成：{} 至 {} 共 {} 个交易日，用时 {}；净值 {:.4f}（{:+.2%}），基准 {}，"
        "最大回撤 {:.2%}，手续费 {:,.0f}；订单 {}，全部成交 {}，拒单 {}",
        portfolios.tradeDate.iloc[0], portfolios.tradeDate.iloc[-1], len(portfolios),
        duration(seconds), net_value.iloc[-1], net_value.iloc[-1] - 1, benchmark, drawdown,
        portfolios.totalFee.iloc[-1], orders, filled, rejected.orderId.nunique(),
    )
    if not rejected.empty:
        rejected = rejected.drop_duplicates("orderId")
        buys = int(rejected.direction.isin([Direction.BUY_OPEN, Direction.BUY_CLOSE]).sum())
        reasons = rejected.orderInfo.dropna().astype(str).str.strip()
        reasons = reasons[reasons != ""].value_counts().head(3)
        log.warning(
            "拒单 {} 笔（买入 {}，卖出 {}）：{}", len(rejected), buys, len(rejected) - buys,
            "；".join(f"{reason} {count} 笔" for reason, count in reasons.items())
            if not reasons.empty else "插件未返回原因，DEBUG 日志含逐笔明细",
        )


def _run(
    algos: Sequence[Algo[Any, C]],
    ctx: C,
    parameters: A,
    settings: DolphinSettings,
    engine_type: type[ResearchBacktest[C]],
) -> BacktestResult[A]:
    with engine_type(settings, parameters, ctx, algos) as engine:
        engine.run(np.timedelta64(parameters.batch_days, "D"))
        portfolios = records_frame(engine.get_daily_total_portfolios(), DailyPortfolio)
        if parameters.benchmark and not portfolios.empty:
            with query(
                {
                    "start_date": parameters.start.isoformat(),
                    "end_date": (parameters.end - timedelta(days=1)).isoformat(),
                    "codes": [source_symbol(parameters.benchmark)],
                    "factors": ["close"],
                }
            ) as result:
                benchmark = result.data
            if benchmark.empty:
                log.warning("基准 {} 没有行情，不输出基准列", parameters.benchmark)
            else:
                benchmark["tradeDate"] = benchmark.time.dt.date
                portfolios = portfolios.merge(
                    benchmark[["tradeDate", "close"]].rename(columns={"close": "benchmarkClosePrice"}),
                    on="tradeDate",
                    how="left",
                    validate="one_to_one",
                )
                prices = portfolios.benchmarkClosePrice.where(
                    np.isfinite(portfolios.benchmarkClosePrice) & (portfolios.benchmarkClosePrice > 0)
                )
                portfolios["benchmarkClosePrice"] = prices
                valid_prices = prices.dropna()
                portfolios["benchmarkNetValue"] = (
                    prices / valid_prices.iloc[0] if not valid_prices.empty else np.nan
                )
        diagnostics = [row for algo in algos for row in algo.diagnostics]
        return BacktestResult(
            trade_details=records_frame(engine.get_trade_details(), TradeDetail),
            daily_positions=records_frame(engine.get_daily_position(), DailyPosition),
            daily_portfolios=portfolios,
            daily_trading_statistics=records_frame(
                engine.get_daily_trading_statistics(), DailyTradingStatistics
            ),
            allocation_diagnostics=pd.DataFrame(diagnostics, columns=["time", "method", "reason"]),
            parameters=parameters.model_copy(deep=True),
        )
