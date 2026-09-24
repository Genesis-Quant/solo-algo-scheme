from collections.abc import Sequence
from datetime import timedelta
from functools import cached_property
from importlib.resources import files
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel

from scheme import (
    Algo,
    Backtest,
    DailyPortfolio,
    DailyPosition,
    DailyTradingStatistics,
    DosVar,
    ResearchContext,
    TradeDetail,
)
from scheme.config import DolphinSettings
from scheme.data.dolphindb import query
from scheme.symbols import engine_symbol, source_symbol
from scheme.utils.packages import validate_context

from .result import BacktestResult
from .schema import BacktestParameters


class ResearchBacktest[C: BaseModel](Backtest[C]):
    def __init__(
        self,
        settings: DolphinSettings,
        parameters: BacktestParameters,
        ctx: C,
        algos: Sequence[Algo[Any, C]],
    ) -> None:
        self.settings = settings
        self.parameters = parameters
        config = {
            "strategyGroup": "stock",
            "cash": 1_000_000.0,
            "commission": 0.0003,
            "tax": 0.0005,
            "latency": np.int32(0),
            "enableSellCloseRestrict": True,
            **parameters.config,
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
        for key in ("cash", "commission", "tax"):
            config[key] = float(config[key])
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
        return self.parameters.universe.evaluate(self.parameters.start, self.parameters.end)


def records_frame(records: Sequence[BaseModel], model: type[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump() for row in records], columns=list(model.model_fields))


def run_backtest[C: ResearchContext[Any]](
    algos: Sequence[Algo[Any, C]],
    ctx: C,
    parameters: BacktestParameters,
    *,
    settings: DolphinSettings | None = None,
    engine_type: type[ResearchBacktest[C]] = ResearchBacktest,
) -> BacktestResult:
    """逆序事件回调后正序消费 Context 消息；仍允许 Algo 直接下单。"""
    settings = settings or DolphinSettings.from_env()
    if not algos:
        raise ValueError("至少需要一个 Algo")
    for algo in algos:
        if not isinstance(algo, Algo):
            raise TypeError("算法必须继承当前 scheme 的 Algo")
        validate_context(algo, ctx)
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
        )
