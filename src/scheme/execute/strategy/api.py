from collections.abc import Sequence
from datetime import timedelta
from functools import cached_property
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
from query.utils import logger as log

from scheme.base import Algo, ResearchContext, StrategyAnalysisParams
from scheme.config import DolphinSettings
from scheme.data import query
from scheme.data.dolphindb.database.session import download, release, upload
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol
from scheme.errors import error_text
from scheme.execute.checkpoints import Checkpoints
from scheme.execute.strategy.assembly import validate_context
from scheme.execute.strategy.engine import Backtest

from .result import BacktestResult

BacktestC = TypeVar("BacktestC", bound=BaseModel, infer_variance=True)
C = TypeVar("C", bound=ResearchContext[Any])
A = TypeVar("A", bound=StrategyAnalysisParams)

MARKET_SCRIPT = """
def dailyMessages(daily, request) {
    valid = select * from daily
        where !isNull(open), !isNull(close), !isNull(pre_close), !isNull(up_limit), !isNull(down_limit),
              open > 0, close > 0, pre_close > 0, up_limit > 0, down_limit > 0
    if (valid.rows() == 0) return table(array(TIMESTAMP, 0) as timestamp)
    openRows = select
            strReplace(strReplace(string(code), ".SH", ".XSHG"), ".SZ", ".XSHE") as code,
            temporalAdd(temporalAdd(timestamp(time), 9, "h"), 30, "m") as timestamp,
            open as lastPrice,
            pre_close as prevClosePrice,
            up_limit as upLimitPrice,
            down_limit as downLimitPrice
        from valid
    closeRows = select
            strReplace(strReplace(string(code), ".SH", ".XSHG"), ".SZ", ".XSHE") as code,
            temporalAdd(timestamp(time), 15, "h") as timestamp,
            close as lastPrice,
            pre_close as prevClosePrice,
            up_limit as upLimitPrice,
            down_limit as downLimitPrice
        from valid
    combined = unionAll(openRows, closeRows)
    rows = select * from combined where timestamp >= request.start, timestamp < request.end order by timestamp, code
    if (rows.rows() == 0) return table(array(TIMESTAMP, 0) as timestamp)
    quantities = take(1000000000l, rows.rows())
    ends = int(1 .. rows.rows())
    return table(
        string(rows.code) as symbol,
        iif(endsWith(string(rows.code), ".XSHE"), "XSHE", "XSHG") as symbolSource,
        rows.timestamp,
        double(rows.lastPrice) as lastPrice,
        double(rows.prevClosePrice) as prevClosePrice,
        double(rows.upLimitPrice) as upLimitPrice,
        double(rows.downLimitPrice) as downLimitPrice,
        quantities as totalBidQty,
        quantities as totalOfferQty,
        arrayVector(ends, double(rows.lastPrice)) as bidPrice,
        arrayVector(ends, double(rows.lastPrice)) as offerPrice,
        arrayVector(ends, quantities) as bidQty,
        arrayVector(ends, quantities) as offerQty
    )
}

def snapshotMessages(reference, request) {
    sourceTable = loadTable(request.database, request.table)
    startTime = timestamp(request.start)
    endTime = timestamp(request.end)
    startDate = date(startTime)
    endDate = date(request.queryEnd)
    raw = select Market, SecurityID, TradeDate, TradeTime, LastPrice,
            BidPrice1, BidPrice2, BidPrice3, BidPrice4, BidPrice5,
            AskPrice1, AskPrice2, AskPrice3, AskPrice4, AskPrice5,
            BidVolume1, BidVolume2, BidVolume3, BidVolume4, BidVolume5,
            AskVolume1, AskVolume2, AskVolume3, AskVolume4, AskVolume5
        from sourceTable
        where TradeDate >= startDate, TradeDate < endDate,
              ((Market == "SH" and SecurityID in request.sh) or
               (Market == "SZ" and SecurityID in request.sz)), LastPrice > 0,
              (TradeTime between 09:30:00.000:11:30:00.000 or
               TradeTime between 13:00:00.000:15:00:00.000)
    timestamps = concatDateTime(raw.TradeDate, raw.TradeTime)
    raw = raw[timestamps >= startTime && timestamps < endTime]
    if (raw.rows() == 0) return table(array(TIMESTAMP, 0) as timestamp)
    quotes = table(
        string(raw.SecurityID) + iif(string(raw.Market) == "SZ", ".XSHE", ".XSHG") as symbol,
        iif(string(raw.Market) == "SZ", "XSHE", "XSHG") as symbolSource,
        raw.TradeDate as tradeDate,
        concatDateTime(raw.TradeDate, raw.TradeTime) as timestamp,
        double(raw.LastPrice) as lastPrice,
        fixedLengthArrayVector(
            double(raw.BidPrice1),
            double(raw.BidPrice2),
            double(raw.BidPrice3),
            double(raw.BidPrice4),
            double(raw.BidPrice5)
        ) as bidPrice,
        fixedLengthArrayVector(
            double(raw.AskPrice1),
            double(raw.AskPrice2),
            double(raw.AskPrice3),
            double(raw.AskPrice4),
            double(raw.AskPrice5)
        ) as offerPrice,
        fixedLengthArrayVector(
            long(raw.BidVolume1),
            long(raw.BidVolume2),
            long(raw.BidVolume3),
            long(raw.BidVolume4),
            long(raw.BidVolume5)
        ) as bidQty,
        fixedLengthArrayVector(
            long(raw.AskVolume1),
            long(raw.AskVolume2),
            long(raw.AskVolume3),
            long(raw.AskVolume4),
            long(raw.AskVolume5)
        ) as offerQty,
        long(raw.BidVolume1 + raw.BidVolume2 + raw.BidVolume3 + raw.BidVolume4 + raw.BidVolume5) as totalBidQty,
        long(raw.AskVolume1 + raw.AskVolume2 + raw.AskVolume3 + raw.AskVolume4 + raw.AskVolume5) as totalOfferQty
    )
    quotes = select * from quotes where symbol in request.symbols
    refs = select
            date(time) as tradeDate,
            strReplace(strReplace(string(code), ".SH", ".XSHG"), ".SZ", ".XSHE") as symbol,
            double(pre_close) as prevClosePrice,
            double(up_limit) as upLimitPrice,
            double(down_limit) as downLimitPrice
        from reference
    quotes = lj(quotes, refs, `tradeDate`symbol)
    if (any(isNull(quotes.prevClosePrice) || isNull(quotes.upLimitPrice) || isNull(quotes.downLimitPrice)))
        throw "五档行情缺少昨收价或涨跌停参考价"
    return select symbol, symbolSource, timestamp, lastPrice, prevClosePrice,
        upLimitPrice, downLimitPrice, totalBidQty, totalOfferQty,
        bidPrice, offerPrice, bidQty, offerQty from quotes order by timestamp, symbol
}
"""


def costs(config: dict[str, Any]) -> dict[str, float]:
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
        self.parameters = parameters
        config = {
            "strategyGroup": "stock",
            "latency": np.int32(0),
            "enableSellCloseRestrict": True,
            **parameters.config,
            **costs(parameters.config),
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
        self.loaded_rows = 0
        self.loaded_dates: set[object] = set()
        self.loaded_daily_batches: set[tuple[np.datetime64, np.datetime64]] = set()
        super().__init__(
            settings,
            engine_name=uuid4().hex,
            config=config,
            ctx=ctx,
            algos=algos,
        )
        try:
            self.symbols = (
                list(parameters.symbols)
                if parameters.symbols is not None
                else self.universe.columns[self.universe.any(axis=0)].tolist()
            )
            self.daily_timeline: np.ndarray | None = None
        except BaseException:
            try:
                self.close()
            except Exception as error:
                log.warning(f"[回测] 初始化失败后关闭引擎异常：{error_text(error)}")
            raise

    def load_timeline(
        self, message: DosVar, start: np.datetime64, end: np.datetime64
    ) -> Sequence[np.datetime64]:
        if self.daily_timeline is None:
            return super().load_timeline(message, start, end)
        return self.daily_timeline[(self.daily_timeline >= start) & (self.daily_timeline < end)]

    def prepare_daily_messages(self) -> DosVar:
        start = np.datetime64(self.parameters.start, "ms")
        end = np.datetime64(self.parameters.end, "ms")
        if not self.symbols:
            self.session.run("soloDailyMessages = table(array(TIMESTAMP, 0) as timestamp); NULL")
            self.daily_timeline = np.array([], dtype="datetime64[ms]")
            self.daily_counts = np.array([], dtype=np.int64)
            return DosVar("soloDailyMessages")
        with query(
            {
                "start_date": self.parameters.start.isoformat(),
                "end_date": (self.parameters.end - timedelta(days=1)).isoformat(),
                "codes": [source_symbol(code) for code in self.symbols],
                "factors": ["open", "close", "pre_close", "up_limit", "down_limit"],
            },
            session=self.session,
            settings=self.settings,
        ) as result:
            self.session.run(MARKET_SCRIPT)
            timeline = self.session.run(
                f"soloDailyMessages = dailyMessages({result.data_ref}, "
                f"dict(`start`end, [timestamp({int(start.astype('int64'))}), "
                f"timestamp({int(end.astype('int64'))})])); "
                "select count(*) as quotes from soloDailyMessages group by timestamp order by timestamp"
            )
        self.daily_timeline = np.asarray(timeline["timestamp"], dtype="datetime64[ms]")
        self.daily_counts = np.asarray(timeline["quotes"], dtype=np.int64)
        return DosVar("soloDailyMessages")

    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        if self.parameters.market_data == "stock_daily":
            if self.daily_timeline is None:
                self.prepare_daily_messages()
            batch = (start, end)
            if batch not in self.loaded_daily_batches:
                mask = (self.daily_timeline >= start) & (self.daily_timeline < end)
                self.loaded_rows += int(self.daily_counts[mask].sum())
                self.loaded_dates.update(pd.DatetimeIndex(self.daily_timeline[mask]).date)
                self.loaded_daily_batches.add(batch)
            self.session.run(
                "soloMessages = select * from soloDailyMessages "
                f"where timestamp >= timestamp({int(start.astype('int64'))}l), "
                f"timestamp < timestamp({int(end.astype('int64'))}l); NULL"
            )
            return DosVar("soloMessages")
        symbols = [engine_symbol(symbol) for symbol in self.symbols]
        if not symbols:
            self.session.run("soloMessages = table(array(TIMESTAMP, 0) as timestamp); NULL")
            return DosVar("soloMessages")
        query_end = end.astype("datetime64[D]")
        if end > query_end:
            query_end += np.timedelta64(1, "D")
        fields = ["pre_close", "up_limit", "down_limit"]
        if self.parameters.market_data == "stock_daily":
            fields = ["open", "close", *fields]
        request = {
            "start_date": str(start.astype("datetime64[D]")),
            "end_date": str(query_end - np.timedelta64(1, "D")),
            # 非交易日批次仍可读取完整交易日历，最终结果只保留本批次。
            "lookback": "P30D",
            "codes": [source_symbol(symbol) for symbol in symbols],
            "factors": fields,
        }
        try:
            with query(
                request,
                session=self.session,
                settings=self.settings,
            ) as result:
                self.session.run(MARKET_SCRIPT)
                if self.parameters.market_data == "stock_snapshot":
                    upload(
                        self.session,
                        {
                            "snapshotRequest": {
                                "database": self.settings.snapshot_database,
                                "table": self.settings.snapshot_table,
                                "start": start,
                                "end": end,
                                "queryEnd": query_end,
                                "symbols": np.asarray(symbols, dtype=str),
                                "sh": np.asarray(
                                    [s.split(".")[0] for s in symbols if s.endswith(".XSHG")],
                                    dtype=str,
                                ),
                                "sz": np.asarray(
                                    [s.split(".")[0] for s in symbols if s.endswith(".XSHE")],
                                    dtype=str,
                                ),
                            },
                        },
                        "五档行情查询条件",
                    )
                    self.session.run(
                        f"soloMessages = snapshotMessages({result.data_ref}, snapshotRequest); NULL"
                    )
                else:
                    upload(
                        self.session,
                        {"marketWindow": {"start": start, "end": end}},
                        "行情回放时间范围",
                    )
                    self.session.run(
                        f"soloMessages = dailyMessages({result.data_ref}, marketWindow); NULL"
                    )
        finally:
            release(
                self.session,
                ["marketWindow", "snapshotRequest"],
                ["dailyMessages", "snapshotMessages"],
            )
        message_rows = int(download(self.session, "soloMessages.rows()", "本批行情记录数"))
        days = download(
            self.session, "exec distinct date(timestamp) from soloMessages", "本批行情日期"
        )
        self.loaded_dates.update(pd.DatetimeIndex(days).date)
        self.loaded_rows += message_rows
        return DosVar("soloMessages")

    @cached_property
    def universe(self) -> pd.DataFrame:
        return self.parameters.universe.evaluate(
            self.parameters.start,
            self.parameters.end,
            session=self.session,
            settings=self.settings,
        )


def records_frame(records: Sequence[BaseModel], model: type[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame(
        [record.model_dump() for record in records], columns=list(model.model_fields)
    )


def run_backtest(
    algos: Sequence[Algo[Any, C]],
    ctx: C,
    parameters: A,
    *,
    settings: DolphinSettings | None = None,
    engine_type: type[ResearchBacktest[C]] = ResearchBacktest,
    checkpoints: Checkpoints | None = None,
) -> BacktestResult[A]:
    settings = settings if settings is not None else DolphinSettings.from_env()
    if not algos:
        raise ValueError("至少需要一个 Algo")
    for algo in algos:
        if not isinstance(algo, Algo):
            raise TypeError("算法必须继承当前 scheme 的 Algo")
        validate_context(algo, ctx)
    config = costs(parameters.config)
    algorithms = "、".join(type(algo).__name__ for algo in algos)
    log.info(
        f"[回测] 开始：{parameters.start} 至 {parameters.end}（不含），{parameters.market_data}，"
        f"每批 {parameters.batch_days} 天；资金 {config['cash']:,.0f}，佣金 {config['commission']:.4%}，"
        f"印花税 {config['tax']:.4%}，基准={parameters.benchmark or '无'}，算法={algorithms}"
    )
    try:
        result = execute_backtest(algos, ctx, parameters, settings, engine_type, checkpoints)
    except Exception as error:
        log.error(
            f"[回测] 失败：{parameters.start} 至 {parameters.end}，"
            f"{error_text(error, settings.password.get_secret_value())}"
        )
        raise
    try:
        summarize(result)
    except Exception as error:
        log.warning(f"[回测] 结果摘要生成失败，原始报告保留：{error_text(error)}")
    return result


def summarize(result: BacktestResult[Any]) -> None:
    portfolios = result.daily_portfolios
    if portfolios.empty:
        log.warning("[回测] 没有生成账户记录")
        return
    net_value = portfolios.netValue
    drawdown = (1 - net_value / net_value.cummax().clip(lower=1.0)).max()
    benchmark = (
        f"{portfolios.benchmarkNetValue.iloc[-1] - 1:+.2%}"
        if "benchmarkNetValue" in portfolios
        else "无"
    )
    details = result.trade_details
    orders = details.orderId.nunique()
    filled = details.loc[details.orderStatus == OrderStatus.FILLED, "orderId"].nunique()
    rejected = details[details.orderStatus == OrderStatus.REJECTED]
    log.info(
        f"[回测] 完成：{portfolios.tradeDate.iloc[0]} 至 {portfolios.tradeDate.iloc[-1]}，"
        f"共 {len(portfolios)} 个交易日；净值 {net_value.iloc[-1]:.4f}（{net_value.iloc[-1] - 1:+.2%}），"
        f"基准 {benchmark}，最大回撤 {drawdown:.2%}，手续费 {portfolios.totalFee.iloc[-1]:,.0f}；"
        f"订单 {orders}，全部成交 {filled}，拒单 {rejected.orderId.nunique()}"
    )
    if not rejected.empty:
        rejected = rejected.drop_duplicates("orderId")
        buys = int(rejected.direction.isin([Direction.BUY_OPEN, Direction.BUY_CLOSE]).sum())
        reasons = rejected.orderInfo.dropna().astype(str).str.strip()
        reasons = reasons[reasons != ""].value_counts().head(3)
        description = (
            "；".join(f"{reason} {count} 笔" for reason, count in reasons.items())
            if not reasons.empty
            else "插件未返回拒单原因；按订单号查看回报或成交明细"
        )
        log.warning(
            f"拒单 {len(rejected)} 笔（买入 {buys}，卖出 {len(rejected) - buys}）：{description}"
        )


RECORDS = frozenset(
    {
        "trade_details",
        "daily_positions",
        "daily_portfolios",
        "daily_trading_statistics",
        "allocation_diagnostics",
    }
)


def with_benchmark(
    portfolios: pd.DataFrame,
    parameters: StrategyAnalysisParams,
    settings: DolphinSettings,
    session: Any | None,
) -> pd.DataFrame:
    """session 为空时由查询自行建立并关闭连接，用于从回放检查点恢复。"""
    if not parameters.benchmark or portfolios.empty:
        return portfolios
    with query(
        {
            "start_date": parameters.start.isoformat(),
            "end_date": (parameters.end - timedelta(days=1)).isoformat(),
            "codes": [source_symbol(parameters.benchmark)],
            "factors": ["close"],
        },
        session=session,
        settings=settings,
    ) as result:
        benchmark = result.data
    if benchmark.empty:
        log.warning(f"基准 {parameters.benchmark} 没有行情，不输出基准列")
        return portfolios
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
    return portfolios


def execute_backtest(
    algos: Sequence[Algo[Any, C]],
    ctx: C,
    parameters: A,
    settings: DolphinSettings,
    engine_type: type[ResearchBacktest[C]],
    checkpoints: Checkpoints | None = None,
) -> BacktestResult[A]:
    # 引擎状态保存在 DolphinDB 回测插件中，回放只能整体复用，不能从中途批次继续。
    records = checkpoints.load("backtest") if checkpoints is not None else None
    if records is not None and not (isinstance(records, dict) and RECORDS <= records.keys()):
        log.warning("[检查点] 回测回放结果格式不符，重新回放")
        records = None
    if records is None:
        with engine_type(settings, parameters, ctx, algos) as engine:
            engine.run(np.timedelta64(parameters.batch_days, "D"))
            diagnostics = [row for algo in algos for row in algo.diagnostics]
            records = {
                "trade_details": records_frame(engine.get_trade_details(), TradeDetail),
                "daily_positions": records_frame(engine.get_daily_position(), DailyPosition),
                "daily_portfolios": records_frame(
                    engine.get_daily_total_portfolios(), DailyPortfolio
                ),
                "daily_trading_statistics": records_frame(
                    engine.get_daily_trading_statistics(), DailyTradingStatistics
                ),
                "allocation_diagnostics": pd.DataFrame(
                    diagnostics, columns=["time", "method", "reason"]
                ),
            }
            if checkpoints is not None:
                checkpoints.save("backtest", records)
            portfolios = with_benchmark(
                records["daily_portfolios"], parameters, settings, engine.session
            )
    else:
        portfolios = with_benchmark(records["daily_portfolios"], parameters, settings, None)
    return BacktestResult(
        trade_details=records["trade_details"],
        daily_positions=records["daily_positions"],
        daily_portfolios=portfolios,
        daily_trading_statistics=records["daily_trading_statistics"],
        allocation_diagnostics=records["allocation_diagnostics"],
        parameters=parameters.model_copy(deep=True),
    )
