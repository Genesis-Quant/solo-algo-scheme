from time import perf_counter
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from scheme.base import Factor, FactorAnalysisParams, FactorParams
from scheme.config import DolphinSettings
from scheme.data import query
from scheme.data.dolphindb.database import create_session
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol
from scheme.log import get_logger

from .result import FactorAnalysisResult

log = get_logger("scheme.factor")

P = TypeVar("P", bound=FactorParams)
A = TypeVar("A", bound=FactorAnalysisParams)


def prepare_panel(panel: pd.DataFrame, parameters: FactorAnalysisParams) -> pd.DataFrame:
    if list(panel.index.names) != ["date", "symbol"] or not panel.index.is_unique:
        raise ValueError("因子面板必须使用唯一的 (date, symbol) 索引")
    frame = (
        panel[parameters.columns].reset_index().rename(columns={"date": "time", "symbol": "code"})
    )
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


def _return_derivatives(max_lag: int) -> dict[str, Any]:
    def shifted(column: str, lag: int) -> str | dict[str, Any]:
        if lag == 0:
            return column
        return {
            "type": "TS", "op": "unary.shift",
            "fields": {"col": column}, "params": {"periods": -lag},
        }

    derivatives: dict[str, Any] = {
        f"adjusted_{price}": {
            "type": "DIRECT", "op": "binary.mul",
            "fields": {"left": price, "right": "adj_factor"},
        }
        for price in ("open", "close")
    }
    for lag in range(1, max_lag + 1):
        for name, numerator, denominator in (
            (f"r{lag - 1}o", shifted("adjusted_open", lag), shifted("adjusted_close", lag - 1)),
            (f"r{lag}", shifted("adjusted_close", lag), shifted("adjusted_open", lag)),
        ):
            derivatives[name] = {
                "type": "DIRECT", "op": "binary.sub",
                "fields": {
                    "left": {
                        "type": "DIRECT", "op": "binary.div",
                        "fields": {"left": numerator, "right": denominator},
                    },
                    "right": 1.0,
                },
            }
    return derivatives


def analyze_factors(
    factor: Factor[P],
    parameters: A,
    *,
    settings: DolphinSettings | None = None,
) -> FactorAnalysisResult[A]:
    settings = settings or DolphinSettings.from_env()
    started = perf_counter()
    log.info(
        "因子分析：{} {} 至 {}（含），因子列 {}，最大滞后 {}，隔夜/日内收益 {}，{} 组，{}",
        type(factor).__name__, parameters.start, parameters.end, ", ".join(parameters.columns),
        parameters.max_lag, ", ".join(parameters.return_columns), parameters.groups,
        "市值加权" if parameters.weight == "market_value" else "等权",
    )
    source = prepare_panel(factor.compute(parameters.start, parameters.end), parameters)
    log.info(
        "因子计算完成：{} 条记录，{} 只证券，{} 个交易日，用时 {:.1f} 秒",
        len(source), source.code.nunique(), source.time.nunique(), perf_counter() - started,
    )
    with create_session(settings) as session:
        # 不扩展研究结束日；滞后价格按查询交易日轴位移，缺行情和超出区间均保留 NULL。
        with query(
            {
                "start_date": parameters.start.isoformat(),
                "end_date": parameters.end.isoformat(),
                "codes": sorted(
                    {source_symbol(s) for s in source.code}
                    | {source_symbol(parameters.calendar_symbol)}
                ),
                "factors": ["circ_mv"] if parameters.weight == "market_value" else [],
                "derivatives": _return_derivatives(parameters.max_lag),
            },
            session=session,
        ) as result:
            codes = sorted(set(source.code) | {engine_symbol(parameters.calendar_symbol)})
            session.upload(
                {
                    "soloFactorSource": source,
                    "soloFactors": np.asarray(parameters.columns, dtype=str),
                    "soloReturns": np.asarray(parameters.return_columns, dtype=str),
                    "soloPeriods": np.arange(1, parameters.max_lag + 1, dtype=np.int32),
                    "soloGroups": np.int32(parameters.groups),
                    "soloSelect": np.int32(parameters.n_select),
                    "soloMarketWeight": parameters.weight == "market_value",
                    "soloSourceCodes": np.asarray([source_symbol(s) for s in codes]),
                    "soloEngineCodes": np.asarray(codes),
                    "soloCalendarSymbol": source_symbol(parameters.calendar_symbol),
                    "soloStart": np.datetime64(parameters.start, "ms"),
                    # 底层执行统计按半开区间过滤；仅统计上界转换为末日次日。
                    "soloEnd": np.datetime64(parameters.end, "ms") + np.timedelta64(1, "D"),
                }
            )
            session.run(f"""
                use factor
                soloCalendar = exec time from {result.data_ref} where code = soloCalendarSymbol
                if (size(soloCalendar) == 0) throw "缺少交易日历基准行情"
                soloPrices = select * from {result.data_ref} where time in soloCalendar
                soloPrices.dropColumns!(`adjusted_open`adjusted_close)
                replaceColumn!(soloPrices, `time, timestamp(soloPrices.time))
                replaceColumn!(soloFactorSource, `time, timestamp(soloFactorSource.time))
                replaceColumn!(soloPrices, `code,
                    dict(soloSourceCodes, soloEngineCodes)[string(soloPrices.code)])
                soloFactorInput = lj(soloFactorSource, soloPrices, `time`code)
                if (!soloMarketWeight) {{
                    soloFactorInput[`circ_mv] = take(1.0, soloFactorInput.rows())
                }}
                soloFactorProcessed = factor::factorFilterNulls(soloFactorInput, soloFactors)
                if (soloFactorProcessed.rows() == 0) throw "没有共同有效的因子样本"
                // 只有参与分组的有效因子记录需要市值权重。
                if (soloMarketWeight) {{
                    if (any(isNull(soloFactorProcessed.circ_mv) || soloFactorProcessed.circ_mv <= 0))
                        throw "市值加权需要每条有效因子记录具备正数 circ_mv"
                }}
                for (factorCol in soloFactors) {{
                    soloRanks = <select int(floor(
                        rank(_$factorCol, true, , true, `first, false) * soloGroups
                        / double(count(_$factorCol))
                    )) as factor_group from soloFactorProcessed context by time>.eval()
                    soloFactorProcessed[factorCol + "_group"] = soloRanks.factor_group
                }}
            """)
            report = FactorAnalysisResult(
                parameters=parameters.model_copy(deep=True),
                processed_data=session.run("soloFactorProcessed"),
                execution_statistics=session.run(
                    "factor::factorExecutionStatistics(soloFactorSource, soloFactorInput, "
                    'soloFactorProcessed, array(STRING, 0), soloStart, soloEnd, "time", "code")'
                ),
                information_coefficient=session.run(
                    'factor::factorInformationCoefficient(soloFactorProcessed, soloReturns, soloFactors, "time")'
                ),
                group_returns=session.run(
                    'factor::factorGroupReturns(soloFactorProcessed, soloReturns, soloFactors, soloGroups, soloSelect, "time", "code", "circ_mv")'
                ),
                group_turnover=session.run(
                    'factor::factorGroupTurnover(soloFactorProcessed, soloFactors, soloPeriods, soloGroups, soloSelect, "time", "code", distinct(soloFactorSource.time))'
                ),
            )
    log.info(
        "因子分析完成：有效样本 {} 条，总用时 {:.1f} 秒",
        len(report.processed_data), perf_counter() - started,
    )
    return report
