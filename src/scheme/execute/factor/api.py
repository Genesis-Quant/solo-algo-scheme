from datetime import timedelta

import numpy as np
import pandas as pd

from scheme.base import Factor, FactorAnalysisParams, FactorParams
from scheme.config import DolphinSettings
from scheme.data import query
from scheme.data.dolphindb.database import create_session
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol

from .result import FactorAnalysisResult


def prepare_panel(panel: pd.DataFrame, parameters: FactorAnalysisParams) -> pd.DataFrame:
    if list(panel.index.names) != ["date", "symbol"] or not panel.index.is_unique:
        raise ValueError("因子面板必须使用唯一的 (date, symbol) 索引")
    frame = (
        panel[parameters.columns].reset_index().rename(columns={"date": "time", "symbol": "code"})
    )
    frame["time"] = pd.to_datetime(frame["time"])
    if (
        frame.empty
        or not frame.time.between(
            pd.Timestamp(parameters.start), pd.Timestamp(parameters.end), inclusive="left"
        ).all()
    ):
        raise ValueError("因子面板为空或包含日期范围外的记录")
    frame["code"] = frame["code"].map(engine_symbol)
    if frame.duplicated(["time", "code"]).any():
        raise ValueError("证券代码规范化后存在重复记录")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in parameters.columns):
        raise TypeError("分析因子必须为数值列")
    frame[parameters.columns] = frame[parameters.columns].replace([np.inf, -np.inf], np.nan)
    return frame.sort_values(["time", "code"])


def analyze_factors[P: FactorParams](
    factor: Factor[P],
    parameters: FactorAnalysisParams,
    *,
    settings: DolphinSettings | None = None,
) -> FactorAnalysisResult:
    settings = settings or DolphinSettings.from_env()
    source = prepare_panel(factor.compute(parameters.start, parameters.end), parameters)
    with create_session(settings) as session:
        # 日历缓冲只用于读取；尾部不足的未来收益保留 NULL，不填零。
        end = parameters.end + timedelta(days=max(parameters.return_periods) * 4 + 30)
        with query(
            {
                "start_date": parameters.start.isoformat(),
                "end_date": (end - timedelta(days=1)).isoformat(),
                "codes": sorted(
                    {source_symbol(s) for s in source.code}
                    | {source_symbol(parameters.calendar_symbol)}
                ),
                "factors": ["circ_mv"] if parameters.weight == "market_value" else [],
                "derivatives": {
                    "adjusted_close": {
                        "type": "DIRECT",
                        "op": "binary.mul",
                        "fields": {"left": "close", "right": "adj_factor"},
                    },
                    **{
                        f"return_{period}": {
                            "type": "DIRECT",
                            "op": "binary.sub",
                            "fields": {
                                "left": {
                                    "type": "DIRECT",
                                    "op": "binary.div",
                                    "fields": {
                                        "left": {
                                            "type": "TS",
                                            "op": "unary.shift",
                                            "fields": {"col": "adjusted_close"},
                                            "params": {"periods": -period},
                                        },
                                        "right": "adjusted_close",
                                    },
                                },
                                "right": 1.0,
                            },
                        }
                        for period in parameters.return_periods
                    },
                },
            },
            session=session,
        ) as result:
            codes = sorted(set(source.code) | {engine_symbol(parameters.calendar_symbol)})
            session.upload(
                {
                    "soloFactorSource": source,
                    "soloFactors": np.asarray(parameters.columns, dtype=str),
                    "soloReturns": np.asarray(
                        [f"return_{p}" for p in parameters.return_periods], dtype=str
                    ),
                    "soloPeriods": np.asarray(parameters.return_periods, dtype=np.int32),
                    "soloGroups": np.int32(parameters.groups),
                    "soloSelect": np.int32(parameters.n_select),
                    "soloMarketWeight": parameters.weight == "market_value",
                    "soloSourceCodes": np.asarray([source_symbol(s) for s in codes]),
                    "soloEngineCodes": np.asarray(codes),
                    "soloCalendarSymbol": source_symbol(parameters.calendar_symbol),
                    "soloStart": np.datetime64(parameters.start, "ms"),
                    "soloEnd": np.datetime64(parameters.end, "ms"),
                }
            )
            session.run(f"""
                use factor
                soloCalendar = exec time from {result.data_ref} where code = soloCalendarSymbol
                if (size(soloCalendar) == 0) throw "缺少交易日历基准行情"
                soloPrices = select * from {result.data_ref} where time in soloCalendar
                soloPrices.dropColumns!(`adjusted_close)
                replaceColumn!(soloPrices, `time, timestamp(soloPrices.time))
                replaceColumn!(soloFactorSource, `time, timestamp(soloFactorSource.time))
                replaceColumn!(soloPrices, `code,
                    dict(soloSourceCodes, soloEngineCodes)[string(soloPrices.code)])
                soloFactorInput = lj(soloFactorSource, soloPrices, `time`code)
                if (soloMarketWeight) {{
                    if (any(isNull(soloFactorInput.circ_mv) || soloFactorInput.circ_mv <= 0))
                        throw "市值加权需要每条因子记录具备正数 circ_mv"
                }} else {{
                    soloFactorInput[`circ_mv] = take(1.0, soloFactorInput.rows())
                }}
                soloFactorProcessed = factor::factorFilterNulls(soloFactorInput, soloFactors)
                if (soloFactorProcessed.rows() == 0) throw "没有共同有效的因子样本"
                for (factorCol in soloFactors) {{
                    soloRanks = <select int(floor(
                        rank(_$factorCol, true, , true, `first, false) * soloGroups
                        / double(count(_$factorCol))
                    )) as factor_group from soloFactorProcessed context by time>.eval()
                    soloFactorProcessed[factorCol + "_group"] = soloRanks.factor_group
                }}
            """)
            return FactorAnalysisResult(
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
