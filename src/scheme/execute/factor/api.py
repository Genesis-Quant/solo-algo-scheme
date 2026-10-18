from time import perf_counter
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from scheme.base import Factor, FactorAnalysisParams, FactorParams
from scheme.base.projects.factor.algo import _validate_result
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
    def shifted(lag: int) -> dict[str, Any]:
        return {
            "type": "TS", "op": "unary.shift",
            "fields": {"col": price_column}, "params": {"periods": -lag},
        }

    return {
        f"P{period}L{lag}": {
            "type": "DIRECT", "op": "binary.sub",
            "fields": {
                "left": {
                    "type": "DIRECT", "op": "binary.div",
                    "fields": {"left": shifted(lag + period), "right": shifted(lag)},
                },
                "right": 1.0,
            },
        }
        for lag in range(1, max_lag + 1)
    }


def analyze_factors(
    factor: Factor[P],
    parameters: A,
    *,
    settings: DolphinSettings | None = None,
) -> FactorAnalysisResult[A]:
    parameters._bind_columns(factor)
    settings = settings or DolphinSettings.from_env()
    started = perf_counter()
    log.info(
        "因子分析：{} {} 至 {}（含），因子列 {}，持有期 {}，价格 {}，最大滞后 {}，收益 {}，{} 组，{}",
        type(factor).__name__, parameters.start, parameters.end, ", ".join(parameters.columns),
        parameters.period, parameters.price_column, parameters.max_lag,
        ", ".join(parameters.return_columns), parameters.groups,
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
                "derivatives": _return_derivatives(
                    parameters.period, parameters.max_lag, parameters.price_column,
                ),
            },
            session=session,
        ) as result:
            codes = sorted(set(source.code) | {engine_symbol(parameters.calendar_symbol)})
            session.upload(
                {
                    "soloFactorSource": source,
                    "soloFactors": np.asarray(parameters.columns, dtype=str),
                    "soloReturns": np.asarray(parameters.return_columns, dtype=str),
                    "soloPeriods": np.asarray([parameters.period], dtype=np.int32),
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
            # 仅聚合中反复执行的三个 helper 保留为会话函数；一次性统计直接生成结果表。
            # 所有顶层临时变量使用 solo 前缀，避免与用户因子列及 SQL 字段混淆。
            session.run(rf"""
                def soloFactorWeightedReturn(weight, ret) {{
                    if (any(!isNull(weight) && weight < 0)) {{
                        throw "市值权重不能为负数"
                    }}
                    valid = !isNull(weight) && !isNull(ret)
                    validWeight = weight[valid]
                    validReturn = ret[valid]
                    totalWeight = sum(validWeight)
                    if (!isValid(totalWeight) || totalWeight <= 0) {{
                        return double(NULL)
                    }}
                    return sum(validWeight * validReturn) \ totalWeight
                }}

                def soloFactorExtremeRanks(values, codes, ascending) {{
                    ranks = take(long(NULL), size(values))
                    if (size(values) == 0) return ranks
                    ranked = table(
                        long(0 .. (size(values) - 1)) as row_id,
                        double(values) as factor_value,
                        string(codes) as factor_code
                    )
                    ranked = select * from ranked where !isNull(factor_value)
                    ranked.sortBy!(`factor_value`factor_code, [ascending, ascending])
                    if (ranked.rows() > 0) {{
                        ranks[ranked.row_id] = long(0 .. (ranked.rows() - 1))
                    }}
                    return ranks
                }}

                def soloFactorExtremeWeightedReturn(
                    factorValue, weight, ret, nSelect, ascending, codes) {{
                    selected = !isNull(factorValue) && (
                        soloFactorExtremeRanks(factorValue, codes, ascending) < int(nSelect)
                    )
                    return soloFactorWeightedReturn(weight[selected], ret[selected])
                }}

                // 行情、日历及共同非空因子样本。
                soloCalendar = exec time from {result.data_ref} where code = soloCalendarSymbol
                if (size(soloCalendar) == 0) throw "缺少交易日历基准行情"
                soloPrices = select * from {result.data_ref} where time in soloCalendar
                replaceColumn!(soloPrices, `time, timestamp(soloPrices.time))
                replaceColumn!(soloFactorSource, `time, timestamp(soloFactorSource.time))
                replaceColumn!(soloPrices, `code,
                    dict(soloSourceCodes, soloEngineCodes)[string(soloPrices.code)])
                soloFactorInput = lj(soloFactorSource, soloPrices, `time`code)
                if (!soloMarketWeight) {{
                    soloFactorInput[`circ_mv] = take(1.0, soloFactorInput.rows())
                }}
                soloInputColumns = soloFactorInput.columnNames()
                for (soloColumn in symbol(soloFactors)) {{
                    if (not(soloColumn in soloInputColumns)) {{
                        throw "required column is missing: " + string(soloColumn)
                    }}
                }}
                soloValidMask = take(true, soloFactorInput.rows())
                for (soloFactorCol in soloFactors) {{
                    soloValidMask = soloValidMask && !isNull(soloFactorInput[soloFactorCol])
                }}
                soloFactorProcessed = soloFactorInput[soloValidMask]
                if (soloFactorProcessed.rows() == 0) throw "没有共同有效的因子样本"
                // 只有参与分组的有效因子记录需要市值权重。
                if (soloMarketWeight) {{
                    if (any(isNull(soloFactorProcessed.circ_mv) || soloFactorProcessed.circ_mv <= 0))
                        throw "市值加权需要每条有效因子记录具备正数 circ_mv"
                }}
                for (soloFactorCol in soloFactors) {{
                    soloRanks = <select int(floor(
                        rank(_$soloFactorCol, true, , true, `first, false) * soloGroups
                        / double(count(_$soloFactorCol))
                    )) as factor_group from soloFactorProcessed context by time>.eval()
                    soloFactorProcessed[soloFactorCol + "_group"] = soloRanks.factor_group
                }}

                // 后续统计共用的列及分组检查，只检查一次。
                if (soloGroups < 2) throw "nGroups must be at least 2"
                if (soloSelect < 1) throw "nSelect must be at least 1"
                soloProcessedColumns = soloFactorProcessed.columnNames()
                soloRequiredColumns = symbol(join(["time", "code", "circ_mv"], soloReturns))
                for (soloColumn in soloRequiredColumns) {{
                    if (not(soloColumn in soloProcessedColumns)) {{
                        throw "required column is missing: " + string(soloColumn)
                    }}
                }}
                for (soloFactorCol in soloFactors) {{
                    soloGroupCol = string(soloFactorCol) + "_group"
                    soloGroupValues = double(soloFactorProcessed[soloGroupCol])
                    soloInvalidGroups = !isNull(soloFactorProcessed[soloFactorCol]) && (
                        isNull(soloGroupValues) || soloGroupValues < 0 ||
                        soloGroupValues >= soloGroups || soloGroupValues != floor(soloGroupValues)
                    )
                    if (any(soloInvalidGroups)) {{
                        soloInvalidSample = soloFactorProcessed[soloInvalidGroups]
                        soloInvalidSample = soloInvalidSample[0:min(5, soloInvalidSample.rows())]
                        throw soloGroupCol + " 必须是 [0, " + string(soloGroups - 1) +
                            "] 内的整数；违规行数=" + string(sum(soloInvalidGroups)) +
                            "，日期=" + string(soloInvalidSample.time) +
                            ", code=" + string(soloInvalidSample.code)
                    }}
                }}

                // 执行统计：此 API 没有额外过滤列，唯一阶段为共同因子非空。
                soloOutputExecutionStatistics = select count(*) as source_count
                    from soloFactorSource
                    where time >= soloStart, time < soloEnd, !isNull(code)
                    group by time
                soloOutputExecutionStatistics.sortBy!(`time)
                soloFilteredCounts = select count(*) as filtered_count
                    from soloFactorProcessed
                    where time >= soloStart, time < soloEnd, !isNull(code)
                    group by time
                soloOutputExecutionStatistics = lj(
                    soloOutputExecutionStatistics, soloFilteredCounts, `time)
                soloOutputExecutionStatistics[`filtered_count] = long(
                    nullFill(soloOutputExecutionStatistics[`filtered_count], 0))
                soloOutputExecutionStatistics["filter0_name"] = take(
                    "因子空值", soloOutputExecutionStatistics.rows())
                soloOutputExecutionStatistics["filter0_count"] =
                    soloOutputExecutionStatistics[`filtered_count]
                soloOutputExecutionStatistics[`retention_rate] = iif(
                    soloOutputExecutionStatistics[`source_count] > 0,
                    double(soloOutputExecutionStatistics[`filtered_count]) \
                        double(soloOutputExecutionStatistics[`source_count]),
                    double(NULL))

                // IC / Rank IC，按时间数值排序，收益 NULL 仍由相关系数聚合处理。
                soloOutputInformationCoefficient = select distinct(time) as time
                    from soloFactorProcessed
                soloOutputInformationCoefficient.sortBy!(`time)
                for (soloFactorCol in soloFactors) {{
                    soloFactorSym = symbol([string(soloFactorCol)])[0]
                    for (soloRetCol in soloReturns) {{
                        soloRetSym = symbol([string(soloRetCol)])[0]
                        soloCorrTable = sql(
                            [
                                sqlCol(`time, , `time),
                                sqlColAlias(makeCall(
                                    corr, sqlCol(soloFactorSym), sqlCol(soloRetSym)), `ic),
                                sqlColAlias(makeCall(
                                    spearmanr, sqlCol(soloFactorSym), sqlCol(soloRetSym)), `rank_ic)
                            ],
                            soloFactorProcessed, , [sqlCol(`time)]
                        ).eval()
                        soloOutputInformationCoefficient = lj(
                            soloOutputInformationCoefficient, soloCorrTable, `time)
                        soloOutputInformationCoefficient.rename!(
                            `ic, symbol([string(soloFactorCol) + "_" + string(soloRetCol) + "_ic"])[0])
                        soloOutputInformationCoefficient.rename!(
                            `rank_ic,
                            symbol([string(soloFactorCol) + "_" + string(soloRetCol) + "_rank_ic"])[0])
                    }}
                }}

                // 分组与独立极端组合收益；极端选择先按因子、证券码排序，再处理收益 NULL。
                soloOutputGroupReturns = select distinct(time) as time from soloFactorProcessed
                soloOutputGroupReturns.sortBy!(`time)
                for (soloFactorCol in soloFactors) {{
                    soloFactorSym = symbol([string(soloFactorCol)])[0]
                    soloGroupCol = string(soloFactorCol) + "_group"
                    soloGroupSym = symbol([soloGroupCol])[0]
                    for (soloRetCol in soloReturns) {{
                        soloRetSym = symbol([string(soloRetCol)])[0]
                        soloExtremes = sql(
                            [
                                sqlCol(`time, , `time),
                                sqlColAlias(makeCall(
                                    soloFactorExtremeWeightedReturn,
                                    sqlCol(soloFactorSym), sqlCol(`circ_mv), sqlCol(soloRetSym),
                                    soloSelect, true, sqlCol(`code)), `bottom_ret),
                                sqlColAlias(makeCall(
                                    soloFactorExtremeWeightedReturn,
                                    sqlCol(soloFactorSym), sqlCol(`circ_mv), sqlCol(soloRetSym),
                                    soloSelect, false, sqlCol(`code)), `top_ret)
                            ],
                            soloFactorProcessed, , [sqlCol(`time)]
                        ).eval()
                        soloBottomRows = select time, bottom_ret as ret from soloExtremes
                        soloOutputGroupReturns = lj(soloOutputGroupReturns, soloBottomRows, `time)
                        soloOutputGroupReturns.rename!(
                            `ret,
                            symbol([string(soloFactorCol) + "_" + string(soloRetCol) + "_bottom"])[0])
                        for (soloGroupId in 0 .. (soloGroups - 1)) {{
                            soloWeighted = sql(
                                [
                                    sqlCol(`time, , `time),
                                    sqlColAlias(makeCall(
                                        soloFactorWeightedReturn,
                                        sqlCol(`circ_mv), sqlCol(soloRetSym)), `ret)
                                ],
                                soloFactorProcessed,
                                expr(sqlCol(soloGroupSym), ==, soloGroupId), [sqlCol(`time)]
                            ).eval()
                            soloOutputGroupReturns = lj(soloOutputGroupReturns, soloWeighted, `time)
                            soloOutputGroupReturns.rename!(
                                `ret, symbol([string(soloFactorCol) + "_" + string(soloRetCol) +
                                    "_group" + string(soloGroupId)])[0])
                        }}
                        soloTopRows = select time, top_ret as ret from soloExtremes
                        soloOutputGroupReturns = lj(soloOutputGroupReturns, soloTopRows, `time)
                        soloOutputGroupReturns.rename!(
                            `ret,
                            symbol([string(soloFactorCol) + "_" + string(soloRetCol) + "_top"])[0])
                    }}
                }}

                // 换手与秩自相关使用完整交易日历，以及唯一持有期 soloPeriods=[period]。
                soloPeriodValues = sort(distinct(int(soloPeriods)))
                if (size(soloPeriodValues) == 0 || any(soloPeriodValues < 1)) {{
                    throw "turnoverPeriods must contain positive integers"
                }}
                soloFactorDates = table(sort(distinct(timestamp(soloCalendar))) as time)
                soloFactorDates.sortBy!(`time)
                soloGroupColumns = "group" + string(0 .. (soloGroups - 1))
                soloOutputGroupTurnover = table(
                    array(TIMESTAMP, 0) as time,
                    array(STRING, 0) as factor,
                    array(INT, 0) as periods,
                    array(DOUBLE, 0) as rank_autocorrelation
                )
                addColumn(soloOutputGroupTurnover, ["bottom"], take(DOUBLE, 1))
                addColumn(soloOutputGroupTurnover, soloGroupColumns, take(DOUBLE, soloGroups))
                addColumn(soloOutputGroupTurnover, ["top"], take(DOUBLE, 1))
                for (soloFactorCol in soloFactors) {{
                    soloGroupCol = string(soloFactorCol) + "_group"
                    soloFactorMembers = <select
                            timestamp(time) as time,
                            string(code) as code,
                            double(_$soloFactorCol) as factor_value
                        from soloFactorProcessed
                        where !isNull(_$soloFactorCol)>.eval()
                    if (soloFactorDates.rows() > 0) {{
                        soloFactorMembers = select distinct time, code, factor_value
                            from soloFactorMembers
                        soloGroupMembers = <select
                                timestamp(time) as time,
                                string(code) as code,
                                int(_$soloGroupCol) as group_id
                            from soloFactorProcessed
                            where !isNull(_$soloFactorCol), !isNull(_$soloGroupCol)>.eval()
                        soloGroupMembers = select distinct time, code, group_id from soloGroupMembers
                        soloRanks = select time, code,
                                rank(factor_value, true, , true, `average, false) as factor_rank
                            from soloFactorMembers context by time
                        soloExtremeRanks = select time, code,
                                soloFactorExtremeRanks(factor_value, code, true) as bottom_rank,
                                soloFactorExtremeRanks(factor_value, code, false) as top_rank
                            from soloFactorMembers context by time
                        soloPortfolioMembers = select time, code, int(group_id + 1) as portfolio_id
                            from soloGroupMembers
                        soloBottomMembers = select time, code, int(0) as portfolio_id
                            from soloExtremeRanks where bottom_rank < int(soloSelect)
                        soloTopMembers = select time, code, int(soloGroups + 1) as portfolio_id
                            from soloExtremeRanks where top_rank < int(soloSelect)
                        soloPortfolioMembers = unionAll(soloPortfolioMembers, soloBottomMembers)
                        soloPortfolioMembers = unionAll(soloPortfolioMembers, soloTopMembers)
                        for (soloPeriod in soloPeriodValues) {{
                            soloRankDateMap = select time, move(time, soloPeriod) as previous_time
                                from soloFactorDates
                            soloCurrentRanks = lj(soloRanks, soloRankDateMap, `time)
                            soloPreviousRanks = select time as previous_time, code,
                                    factor_rank as previous_rank from soloRanks
                            soloRankPairs = lj(
                                soloCurrentRanks, soloPreviousRanks, ["previous_time", "code"])
                            soloRankRows = select corr(factor_rank, previous_rank) as rank_autocorrelation
                                from soloRankPairs where !isNull(previous_rank) group by time
                            soloCurrentMembers = lj(soloPortfolioMembers, soloRankDateMap, `time)
                            soloPreviousMembers = select time as previous_time, code, portfolio_id,
                                    true as retained from soloPortfolioMembers
                            soloMatchedMembers = lj(
                                soloCurrentMembers, soloPreviousMembers,
                                ["previous_time", "code", "portfolio_id"])
                            soloTurnoverRows = select
                                    1.0 - sum(iif(isNull(retained), 0, 1)) \ count(*) as turnover
                                from soloMatchedMembers
                                where !isNull(previous_time) group by time, portfolio_id
                            soloPeriodResult = select time from soloFactorDates
                            soloPeriodResult["factor"] = take(string(soloFactorCol), soloPeriodResult.rows())
                            soloPeriodResult["periods"] = take(int(soloPeriod), soloPeriodResult.rows())
                            soloPeriodResult = lj(soloPeriodResult, soloRankRows, `time)
                            for (soloPortfolioId in 0 .. (soloGroups + 1)) {{
                                soloPortfolioRows = select time, turnover from soloTurnoverRows
                                    where portfolio_id == soloPortfolioId
                                if (soloPortfolioId == 0) {{
                                    soloPortfolioColumn = `bottom
                                }} else if (soloPortfolioId == soloGroups + 1) {{
                                    soloPortfolioColumn = `top
                                }} else {{
                                    soloPortfolioColumn = symbol(["group" + string(soloPortfolioId - 1)])[0]
                                }}
                                soloPortfolioRows.rename!(`turnover, soloPortfolioColumn)
                                soloPeriodResult = lj(soloPeriodResult, soloPortfolioRows, `time)
                            }}
                            soloOutputGroupTurnover = unionAll(soloOutputGroupTurnover, soloPeriodResult)
                        }}
                    }}
                }}
                soloOutputGroupTurnover.sortBy!(`factor`periods`time)
            """)
            report = FactorAnalysisResult(
                parameters=parameters.model_copy(deep=True),
                processed_data=session.run("soloFactorProcessed"),
                execution_statistics=session.run("soloOutputExecutionStatistics"),
                information_coefficient=session.run("soloOutputInformationCoefficient"),
                group_returns=session.run("soloOutputGroupReturns"),
                group_turnover=session.run("soloOutputGroupTurnover"),
            )
    log.info(
        "因子分析完成：有效样本 {} 条，总用时 {:.1f} 秒",
        len(report.processed_data), perf_counter() - started,
    )
    return report
