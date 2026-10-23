from time import perf_counter
from typing import Any, TypeVar
from uuid import uuid4

import numpy as np
import pandas as pd
from runtime.utils import logger as log

from scheme.base import Factor, FactorAnalysisParams, FactorParams
from scheme.base.projects.factor.algo import _validate_result
from scheme.config import DolphinSettings
from scheme.data import query
from scheme.data.dolphindb.database import create_session
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol

from .result import FactorAnalysisResult

P = TypeVar("P", bound=FactorParams)
A = TypeVar("A", bound=FactorAnalysisParams)


def prepare_panel(panel: pd.DataFrame, parameters: FactorAnalysisParams) -> pd.DataFrame:
    panel = _validate_result(panel, parameters.columns, "因子分析")
    frame = panel.reset_index().rename(columns={"date": "time", "symbol": "code"})
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame[frame.time.between(
        pd.Timestamp(parameters.start), pd.Timestamp(parameters.end), inclusive="both"
    )]
    if frame.empty:
        raise ValueError("因子面板为空")
    frame["code"] = frame["code"].map(engine_symbol)
    if frame.duplicated(["time", "code"]).any():
        raise ValueError("证券代码规范化后存在重复记录")
    frame[parameters.columns] = frame[parameters.columns].apply(pd.to_numeric).replace(
        [np.inf, -np.inf], np.nan
    )
    return frame.sort_values(["time", "code"])


def _return_derivatives(period: int, max_lag: int, price_column: str) -> dict[str, Any]:
    def shifted_log(lag: int) -> dict[str, Any]:
        return {
            "type": "DIRECT", "op": "unary.log", "params": {},
            "fields": {"col": {
                "type": "TS", "op": "unary.shift",
                "fields": {"col": price_column}, "params": {"periods": -lag},
            }},
        }

    # 等价于 ln(price[t+lag+period] / price[t+lag])；分别取 log，任一非正端点均为 NULL。
    return {
        f"P{period}L{lag}": {
            "type": "DIRECT", "op": "binary.sub",
            "fields": {"left": shifted_log(lag + period), "right": shifted_log(lag)},
        }
        for lag in range(1, max_lag + 1)
    }


def _calendar_table(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"]).astype("datetime64[ns]")
    if frame["time"].duplicated().any():
        raise ValueError("因子统计返回重复交易日期")
    return frame.set_index("time").reindex(calendar).rename_axis("time").reset_index()


def analyze_factors(
    factor: Factor[P],
    parameters: A,
    *,
    settings: DolphinSettings | None = None,
) -> FactorAnalysisResult[A]:
    parameters._bind_columns(factor)
    settings = settings if settings is not None else DolphinSettings.from_env()
    started = perf_counter()
    log.info(
        "因子分析：{} 至 {}，因子={}，收益={}，分组={}，加权={}；使用 factor 模块",
        parameters.start, parameters.end, parameters.columns,
        parameters.return_columns, parameters.groups, parameters.weight,
    )
    source = prepare_panel(factor.compute(parameters.start, parameters.end), parameters)
    prefix = "soloFa" + uuid4().hex
    names = {key: prefix + key for key in (
        "Source", "Factors", "Returns", "Periods", "Groups", "Select", "MarketWeight",
        "SourceCodes", "EngineCodes", "CalendarSymbol", "Start", "End", "Calendar",
        "Prices", "Input", "Processed",
    )}
    with create_session(settings) as session:
        try:
            with query(
                {
                    "start_date": parameters.start.isoformat(),
                    "end_date": parameters.end.isoformat(),
                    "codes": sorted({source_symbol(s) for s in source.code}
                                    | {source_symbol(parameters.calendar_symbol)}),
                    "factors": ["circ_mv"] if parameters.weight == "market_value" else [],
                    "derivatives": _return_derivatives(
                        parameters.period, parameters.max_lag, parameters.price_column,
                    ),
                },
                session=session, settings=settings,
            ) as result:
                codes = sorted(set(source.code) | {engine_symbol(parameters.calendar_symbol)})
                session.upload({
                    names["Source"]: source,
                    names["Factors"]: np.asarray(parameters.columns, dtype=str),
                    names["Returns"]: np.asarray(parameters.return_columns, dtype=str),
                    names["Periods"]: np.asarray([parameters.period], dtype=np.int32),
                    names["Groups"]: np.int32(parameters.groups),
                    names["Select"]: np.int32(parameters.n_select),
                    names["MarketWeight"]: parameters.weight == "market_value",
                    names["SourceCodes"]: np.asarray([source_symbol(s) for s in codes]),
                    names["EngineCodes"]: np.asarray(codes),
                    names["CalendarSymbol"]: source_symbol(parameters.calendar_symbol),
                    names["Start"]: np.datetime64(parameters.start, "ms"),
                    names["End"]: np.datetime64(parameters.end, "ms") + np.timedelta64(1, "D"),
                })
                session.run(f"""
                    use factor
                    {names['Calendar']} = exec time from {result.data_ref}
                        where code = {names['CalendarSymbol']}
                    if (size({names['Calendar']}) == 0) throw "缺少交易日历基准行情"
                    {names['Prices']} = select * from {result.data_ref}
                        where time in {names['Calendar']}
                    replaceColumn!({names['Prices']}, `time, timestamp({names['Prices']}.time))
                    replaceColumn!({names['Source']}, `time, timestamp({names['Source']}.time))
                    replaceColumn!({names['Prices']}, `code,
                        dict({names['SourceCodes']}, {names['EngineCodes']})[string({names['Prices']}.code)])
                    {names['Input']} = lj({names['Source']}, {names['Prices']}, `time`code)
                    if (!{names['MarketWeight']}) {{
                        {names['Input']}[`circ_mv] = take(1.0, {names['Input']}.rows())
                    }}
                    {names['Processed']} = factor::factorFilterNulls({names['Input']}, {names['Factors']})
                    if ({names['Processed']}.rows() == 0) throw "没有共同有效的因子样本"
                    if ({names['MarketWeight']}) {{
                        if (any(isNull({names['Processed']}.circ_mv) || {names['Processed']}.circ_mv <= 0))
                            throw "市值加权需要每条有效因子记录具备正数 circ_mv"
                    }}
                    for (factorCol in {names['Factors']}) {{
                        ranks = <select int(floor(
                            rank(_$factorCol, true, , true, `first, false) * {names['Groups']}
                            / double(count(_$factorCol))
                        )) as factor_group from {names['Processed']} context by time>.eval()
                        {names['Processed']}[factorCol + "_group"] = ranks.factor_group
                    }}
                """)
                calendar = pd.DatetimeIndex(session.run(names["Calendar"])).unique().sort_values()
                log.info("factor 模块：覆盖统计、IC、分组收益和换手率")
                report = FactorAnalysisResult(
                    parameters=parameters.model_copy(deep=True),
                    processed_data=session.run(names["Processed"]),
                    execution_statistics=session.run(
                        f"factor::factorExecutionStatistics({names['Source']}, {names['Input']}, "
                        f"{names['Processed']}, array(STRING, 0), {names['Start']}, {names['End']}, "
                        '"time", "code")'
                    ),
                    information_coefficient=session.run(
                        f"factor::factorInformationCoefficient({names['Processed']}, "
                        f"{names['Returns']}, {names['Factors']}, \"time\")"
                    ),
                    group_returns=_calendar_table(session.run(
                        f"factor::factorGroupReturns({names['Processed']}, {names['Returns']}, "
                        f"{names['Factors']}, {names['Groups']}, {names['Select']}, "
                        '"time", "code", "circ_mv")'
                    ), calendar),
                    group_turnover=session.run(
                        f"factor::factorGroupTurnover({names['Processed']}, {names['Factors']}, "
                        f"{names['Periods']}, {names['Groups']}, {names['Select']}, "
                        f'"time", "code", {names["Calendar"]})'
                    ),
                )
        finally:
            try:
                session.run("undef(" + repr(list(names.values())).replace("'", '"') + ", VAR)")
            except Exception:
                pass
    log.success("因子分析完成：输入 {} 行，有效 {} 行，用时 {:.1f} 秒",
                len(source), len(report.processed_data), perf_counter() - started)
    return report
