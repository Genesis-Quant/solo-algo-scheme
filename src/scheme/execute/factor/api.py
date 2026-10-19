from typing import Any, TypeVar

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


def analyze_factors(
    factor: Factor[P],
    parameters: A,
    *,
    settings: DolphinSettings | None = None,
) -> FactorAnalysisResult[A]:
    parameters._bind_columns(factor)
    settings = settings if settings is not None else DolphinSettings.from_env()
    log.info(
        "因子分析：{} 至 {}，因子={}，收益={}，分组={}，加权={}",
        parameters.start, parameters.end, parameters.columns,
        parameters.return_columns, parameters.groups, parameters.weight,
    )
    source = prepare_panel(factor.compute(parameters.start, parameters.end), parameters)
    with create_session(settings) as session:
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
                "sourceTable": source,
                "factorColNames": np.asarray(parameters.columns, dtype=str),
                "returnColNames": np.asarray(parameters.return_columns, dtype=str),
                "turnoverPeriods": np.asarray([parameters.period], dtype=np.int32),
                "nGroups": np.int32(parameters.groups),
                "nSelect": np.int32(parameters.n_select),
                "marketWeight": parameters.weight == "market_value",
                "sourceCodes": np.asarray([source_symbol(s) for s in codes]),
                "engineCodes": np.asarray(codes),
                "calendarSymbol": source_symbol(parameters.calendar_symbol),
                "startTime": np.datetime64(parameters.start, "ms"),
                "endTime": np.datetime64(parameters.end, "ms") + np.timedelta64(1, "D"),
            })
            log.info("session.run: 保留共同有效因子并分组")
            session.run(rf"""
                def factorWeightedReturn(weight, ret) {{
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

                def factorExtremeRanks(values, codes, ascending) {{
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

                def factorExtremeWeightedReturn(
                    factorValue, weight, ret, nSelect, ascending, codes) {{
                    selected = !isNull(factorValue) && (
                        factorExtremeRanks(factorValue, codes, ascending) < int(nSelect)
                    )
                    return factorWeightedReturn(weight[selected], ret[selected])
                }}

                // 行情、日历及共同非空因子样本。
                calendar = exec time from {result.data_ref} where code = calendarSymbol
                if (size(calendar) == 0) throw "缺少交易日历基准行情"
                prices = select * from {result.data_ref} where time in calendar
                replaceColumn!(prices, `time, timestamp(prices.time))
                replaceColumn!(sourceTable, `time, timestamp(sourceTable.time))
                replaceColumn!(prices, `code,
                    dict(sourceCodes, engineCodes)[string(prices.code)])
                rawFactorTable = lj(sourceTable, prices, `time`code)
                if (!marketWeight) {{
                    rawFactorTable[`circ_mv] = take(1.0, rawFactorTable.rows())
                }}
                inputColumns = rawFactorTable.columnNames()
                for (column in symbol(factorColNames)) {{
                    if (not(column in inputColumns)) {{
                        throw "required column is missing: " + string(column)
                    }}
                }}
                validMask = take(true, rawFactorTable.rows())
                for (factorCol in factorColNames) {{
                    validMask = validMask && !isNull(rawFactorTable[factorCol])
                }}
                processedTable = rawFactorTable[validMask]
                if (processedTable.rows() == 0) throw "没有共同有效的因子样本"
                // 只有参与分组的有效因子记录需要市值权重。
                if (marketWeight) {{
                    if (any(isNull(processedTable.circ_mv) || processedTable.circ_mv <= 0))
                        throw "市值加权需要每条有效因子记录具备正数 circ_mv"
                }}
                for (factorCol in factorColNames) {{
                    ranks = <select int(floor(
                        rank(_$factorCol, true, , true, `first, false) * nGroups
                        / double(count(_$factorCol))
                    )) as factor_group from processedTable context by time>.eval()
                    processedTable[factorCol + "_group"] = ranks.factor_group
                }}

                // 后续统计共用的列及分组检查，只检查一次。
                if (nGroups < 2) throw "nGroups must be at least 2"
                if (nSelect < 1) throw "nSelect must be at least 1"
                processedColumns = processedTable.columnNames()
                requiredColumns = symbol(join(["time", "code", "circ_mv"], returnColNames))
                for (column in requiredColumns) {{
                    if (not(column in processedColumns)) {{
                        throw "required column is missing: " + string(column)
                    }}
                }}
                for (factorCol in factorColNames) {{
                    groupCol = string(factorCol) + "_group"
                    groupValues = double(processedTable[groupCol])
                    invalidGroups = !isNull(processedTable[factorCol]) && (
                        isNull(groupValues) || groupValues < 0 ||
                        groupValues >= nGroups || groupValues != floor(groupValues)
                    )
                    if (any(invalidGroups)) {{
                        invalidSample = processedTable[invalidGroups]
                        invalidSample = invalidSample[0:min(5, invalidSample.rows())]
                        throw groupCol + " 必须是 [0, " + string(nGroups - 1) +
                            "] 内的整数；违规行数=" + string(sum(invalidGroups)) +
                            "，日期=" + string(invalidSample.time) +
                            ", code=" + string(invalidSample.code)
                    }}
                }}
                NULL
            """)
            log.info("session.run: 统计因子有效覆盖")
            session.run(r"""
                // 执行统计：此 API 没有额外过滤列，唯一阶段为共同因子非空。
                executionStatistics = select count(*) as source_count
                    from sourceTable
                    where time >= startTime, time < endTime, !isNull(code)
                    group by time
                executionStatistics.sortBy!(`time)
                filteredCounts = select count(*) as filtered_count
                    from processedTable
                    where time >= startTime, time < endTime, !isNull(code)
                    group by time
                executionStatistics = lj(
                    executionStatistics, filteredCounts, `time)
                executionStatistics[`filtered_count] = long(
                    nullFill(executionStatistics[`filtered_count], 0))
                executionStatistics["filter0_name"] = take(
                    "因子空值", executionStatistics.rows())
                executionStatistics["filter0_count"] =
                    executionStatistics[`filtered_count]
                executionStatistics[`retention_rate] = iif(
                    executionStatistics[`source_count] > 0,
                    double(executionStatistics[`filtered_count]) \
                        double(executionStatistics[`source_count]),
                    double(NULL))
                NULL
            """)
            log.info("session.run: 计算全部因子的 IC 和 RankIC")
            session.run(r"""
                // IC / Rank IC，按时间数值排序，收益 NULL 仍由相关系数聚合处理。
                informationCoefficient = select distinct(time) as time
                    from processedTable
                informationCoefficient.sortBy!(`time)
                for (factorCol in factorColNames) {
                    factorSym = symbol([string(factorCol)])[0]
                    for (retCol in returnColNames) {
                        retSym = symbol([string(retCol)])[0]
                        corrTable = sql(
                            [
                                sqlCol(`time, , `time),
                                sqlColAlias(makeCall(
                                    corr, sqlCol(factorSym), sqlCol(retSym)), `ic),
                                sqlColAlias(makeCall(
                                    spearmanr, sqlCol(factorSym), sqlCol(retSym)), `rank_ic)
                            ],
                            processedTable, , [sqlCol(`time)]
                        ).eval()
                        informationCoefficient = lj(
                            informationCoefficient, corrTable, `time)
                        informationCoefficient.rename!(
                            `ic, symbol([string(factorCol) + "_" + string(retCol) + "_ic"])[0])
                        informationCoefficient.rename!(
                            `rank_ic,
                            symbol([string(factorCol) + "_" + string(retCol) + "_rank_ic"])[0])
                    }
                }
                NULL
            """)
            log.info("session.run: 计算分组与极端组合对数收益")
            session.run(r"""
                // 分组与独立极端组合对数收益不预平滑；全交易日轴保留无共同因子日期及收益 NULL。
                // 极端选择先按因子、证券码排序，再处理收益 NULL。
                groupReturns = table(sort(distinct(timestamp(calendar))) as time)
                groupReturns.sortBy!(`time)
                for (factorCol in factorColNames) {
                    factorSym = symbol([string(factorCol)])[0]
                    groupCol = string(factorCol) + "_group"
                    groupSym = symbol([groupCol])[0]
                    for (retCol in returnColNames) {
                        retSym = symbol([string(retCol)])[0]
                        extremes = sql(
                            [
                                sqlCol(`time, , `time),
                                sqlColAlias(makeCall(
                                    factorExtremeWeightedReturn,
                                    sqlCol(factorSym), sqlCol(`circ_mv), sqlCol(retSym),
                                    nSelect, true, sqlCol(`code)), `bottom_ret),
                                sqlColAlias(makeCall(
                                    factorExtremeWeightedReturn,
                                    sqlCol(factorSym), sqlCol(`circ_mv), sqlCol(retSym),
                                    nSelect, false, sqlCol(`code)), `top_ret)
                            ],
                            processedTable, , [sqlCol(`time)]
                        ).eval()
                        bottomRows = select time, bottom_ret as ret from extremes
                        groupReturns = lj(groupReturns, bottomRows, `time)
                        groupReturns.rename!(
                            `ret,
                            symbol([string(factorCol) + "_" + string(retCol) + "_bottom"])[0])
                        for (groupId in 0 .. (nGroups - 1)) {
                            weighted = sql(
                                [
                                    sqlCol(`time, , `time),
                                    sqlColAlias(makeCall(
                                        factorWeightedReturn,
                                        sqlCol(`circ_mv), sqlCol(retSym)), `ret)
                                ],
                                processedTable,
                                expr(sqlCol(groupSym), ==, groupId), [sqlCol(`time)]
                            ).eval()
                            groupReturns = lj(groupReturns, weighted, `time)
                            groupReturns.rename!(
                                `ret, symbol([string(factorCol) + "_" + string(retCol) +
                                    "_group" + string(groupId)])[0])
                        }
                        topRows = select time, top_ret as ret from extremes
                        groupReturns = lj(groupReturns, topRows, `time)
                        groupReturns.rename!(
                            `ret,
                            symbol([string(factorCol) + "_" + string(retCol) + "_top"])[0])
                    }
                }
                NULL
            """)
            log.info("session.run: 计算分组换手率与秩自相关")
            session.run(r"""
                // 换手与秩自相关使用完整交易日历，以及唯一持有期 turnoverPeriods=[period]。
                periodValues = sort(distinct(int(turnoverPeriods)))
                if (size(periodValues) == 0 || any(periodValues < 1)) {
                    throw "turnoverPeriods must contain positive integers"
                }
                factorDates = table(sort(distinct(timestamp(calendar))) as time)
                factorDates.sortBy!(`time)
                groupColumns = "group" + string(0 .. (nGroups - 1))
                groupTurnover = table(
                    array(TIMESTAMP, 0) as time,
                    array(STRING, 0) as factor,
                    array(INT, 0) as periods,
                    array(DOUBLE, 0) as rank_autocorrelation
                )
                addColumn(groupTurnover, ["bottom"], take(DOUBLE, 1))
                addColumn(groupTurnover, groupColumns, take(DOUBLE, nGroups))
                addColumn(groupTurnover, ["top"], take(DOUBLE, 1))
                for (factorCol in factorColNames) {
                    groupCol = string(factorCol) + "_group"
                    factorMembers = <select
                            timestamp(time) as time,
                            string(code) as code,
                            double(_$factorCol) as factor_value
                        from processedTable
                        where !isNull(_$factorCol)>.eval()
                    if (factorDates.rows() > 0) {
                        factorMembers = select distinct time, code, factor_value
                            from factorMembers
                        groupMembers = <select
                                timestamp(time) as time,
                                string(code) as code,
                                int(_$groupCol) as group_id
                            from processedTable
                            where !isNull(_$factorCol), !isNull(_$groupCol)>.eval()
                        groupMembers = select distinct time, code, group_id from groupMembers
                        ranks = select time, code,
                                rank(factor_value, true, , true, `average, false) as factor_rank
                            from factorMembers context by time
                        extremeRanks = select time, code,
                                factorExtremeRanks(factor_value, code, true) as bottom_rank,
                                factorExtremeRanks(factor_value, code, false) as top_rank
                            from factorMembers context by time
                        portfolioMembers = select time, code, int(group_id + 1) as portfolio_id
                            from groupMembers
                        bottomMembers = select time, code, int(0) as portfolio_id
                            from extremeRanks where bottom_rank < int(nSelect)
                        topMembers = select time, code, int(nGroups + 1) as portfolio_id
                            from extremeRanks where top_rank < int(nSelect)
                        portfolioMembers = unionAll(portfolioMembers, bottomMembers)
                        portfolioMembers = unionAll(portfolioMembers, topMembers)
                        for (period in periodValues) {
                            rankDateMap = select time, move(time, period) as previous_time
                                from factorDates
                            currentRanks = lj(ranks, rankDateMap, `time)
                            previousRanks = select time as previous_time, code,
                                    factor_rank as previous_rank from ranks
                            rankPairs = lj(
                                currentRanks, previousRanks, ["previous_time", "code"])
                            rankRows = select corr(factor_rank, previous_rank) as rank_autocorrelation
                                from rankPairs where !isNull(previous_rank) group by time
                            currentMembers = lj(portfolioMembers, rankDateMap, `time)
                            previousMembers = select time as previous_time, code, portfolio_id,
                                    true as retained from portfolioMembers
                            matchedMembers = lj(
                                currentMembers, previousMembers,
                                ["previous_time", "code", "portfolio_id"])
                            turnoverRows = select
                                    1.0 - sum(iif(isNull(retained), 0, 1)) \ count(*) as turnover
                                from matchedMembers
                                where !isNull(previous_time) group by time, portfolio_id
                            periodResult = select time from factorDates
                            periodResult["factor"] = take(string(factorCol), periodResult.rows())
                            periodResult["periods"] = take(int(period), periodResult.rows())
                            periodResult = lj(periodResult, rankRows, `time)
                            for (portfolioId in 0 .. (nGroups + 1)) {
                                portfolioRows = select time, turnover from turnoverRows
                                    where portfolio_id == portfolioId
                                if (portfolioId == 0) {
                                    portfolioColumn = `bottom
                                } else if (portfolioId == nGroups + 1) {
                                    portfolioColumn = `top
                                } else {
                                    portfolioColumn = symbol(["group" + string(portfolioId - 1)])[0]
                                }
                                portfolioRows.rename!(`turnover, portfolioColumn)
                                periodResult = lj(periodResult, portfolioRows, `time)
                            }
                            groupTurnover = unionAll(groupTurnover, periodResult)
                        }
                    }
                }
                groupTurnover.sortBy!(`factor`periods`time)
                NULL
            """)
            report = FactorAnalysisResult(
                parameters=parameters.model_copy(deep=True),
                processed_data=session.run("processedTable"),
                execution_statistics=session.run("executionStatistics"),
                information_coefficient=session.run("informationCoefficient"),
                group_returns=session.run("groupReturns"),
                group_turnover=session.run("groupTurnover"),
            )
    log.success("因子分析完成：输入 {} 行，有效 {} 行，{} 个交易日，各收益有效样本={}",
                len(source), len(report.processed_data), len(report.execution_statistics),
                report.processed_data[parameters.return_columns].notna().sum().to_dict())
    if len(report.processed_data) < len(source):
        log.warning("共同有效样本剔除 {} 行，不填补无效因子值", len(source) - len(report.processed_data))
    return report
