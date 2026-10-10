from collections.abc import Sequence
from datetime import timedelta
from functools import cached_property
from importlib.resources import files
from time import perf_counter
from typing import Any
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


def _costs(config: dict[str, Any]) -> dict[str, float]:
    defaults = {"cash": 1_000_000.0, "commission": 0.0003, "tax": 0.0005}
    return {key: float(config.get(key, value)) for key, value in defaults.items()}


class ResearchBacktest[C: BaseModel](Backtest[C]):
    def __init__(
        self,
        settings: DolphinSettings,
        parameters: StrategyAnalysisParams,
        ctx: C,
        algos: Sequence[Algo[Any, C]],
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
            self.session.run(
                files(__package__).joinpath("messages.dos").read_text(encoding="utf-8")
            )
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
            # 停牌、未上市或已退市证券当日没有日线，不能成交，也不合成快照。
            frame = frame[frame[["open", "close", "pre_close"]].notna().any(axis=1)]
            if frame.empty:
                self.session.run("soloMessages = table(array(TIMESTAMP,0) as timestamp)")
                return DosVar("soloMessages")
            columns = ["open", "close", "pre_close", "up_limit", "down_limit"]
            if frame[columns].isna().any().any() or (frame[columns] <= 0).any().any():
                raise ValueError("日线合成快照需要有效 OHLC、昨收与涨跌停价；不推测缺失价格")
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


def run_backtest[C: ResearchContext[Any], A: StrategyAnalysisParams](
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


def _run[C: ResearchContext[Any], A: StrategyAnalysisParams](
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
                raise ValueError("基准数据不可用")
            benchmark["tradeDate"] = benchmark.time.dt.date
            portfolios = portfolios.merge(
                benchmark[["tradeDate", "close"]].rename(columns={"close": "benchmarkClosePrice"}),
                on="tradeDate",
                how="left",
                validate="one_to_one",
            )
            prices = portfolios.benchmarkClosePrice
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError("基准缺少回测交易日或收盘价无效")
            portfolios["benchmarkNetValue"] = prices / prices.iloc[0]
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
