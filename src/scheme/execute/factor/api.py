from typing import TypeVar

import numpy as np
import pandas as pd
from query.utils import logger as log

from scheme.base import Factor, FactorAnalysisParams, FactorParams
from scheme.base.projects.factor.algo import validate_result
from scheme.config import DolphinSettings
from scheme.data import query
from scheme.data.dolphindb.database.session import open_session, release, upload
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol
from scheme.errors import error_text
from scheme.execute.checkpoints import Checkpoints

from .result import FactorAnalysisResult

P = TypeVar("P", bound=FactorParams)
A = TypeVar("A", bound=FactorAnalysisParams)

ANALYSIS_SCRIPT = r"""
def factorStringVector(cols) {
    if (typestr(cols) == "STRING") {
        return [string(cols)]
    }
    return string(cols)
}

def factorCheckColumns(tb, requiredColumns) {
    columns = tb.columnNames()
    for (col in requiredColumns) {
        if (not(col in columns)) {
            throw "required column is missing: " + string(col)
        }
    }
}

def factorFilterNulls(factorTable, factorCols) {
    factorColNames = factorStringVector(factorCols)
    factorCheckColumns(factorTable, symbol(factorColNames))
    validMask = take(true, factorTable.rows())
    for (factorCol in factorColNames) {
        validMask = validMask && !isNull(factorTable[factorCol])
    }
    return factorTable[validMask]
}

def factorExecutionStatistics(
    sourceTable,
    computedTable,
    processedTable,
    filterCols,
    startTime,
    endTime,
    timeCol="time",
    codeCol="code") {
    filterColNames = factorStringVector(filterCols)
    requiredColumns = symbol([timeCol, codeCol])
    factorCheckColumns(sourceTable, requiredColumns)
    factorCheckColumns(computedTable, requiredColumns)
    factorCheckColumns(processedTable, requiredColumns)
    if (size(filterColNames) > 0) {
        factorCheckColumns(computedTable, symbol(filterColNames))
    }

    result =
        <select count(*) as source_count
         from sourceTable
         where
            _$timeCol >= startTime,
            _$timeCol < endTime,
            !isNull(_$codeCol)
         group by _$timeCol>.eval()
    result.rename!(symbol([timeCol])[0], `time)
    result.sortBy!(`time)

    timeSym = symbol([timeCol])[0]
    codeSym = symbol([codeCol])[0]
    computedMask = (
        computedTable[timeSym] >= startTime &&
        computedTable[timeSym] < endTime &&
        !isNull(computedTable[codeSym])
    )
    if (size(filterColNames) > 0) {
        for (index in 0 .. (size(filterColNames) - 1)) {
            filterSym = symbol([string(filterColNames[index])])[0]
            computedMask = computedMask && nullFill(
                computedTable[filterSym],
                false
            )
            stageRows = table(
                computedTable[timeSym][computedMask] as time
            )
            stageCounts = select count(*) as stage_count
                from stageRows
                group by time
            stageColumn = symbol([
                "filter" + string(index) + "_count"
            ])[0]
            stageCounts.rename!(`stage_count, stageColumn)
            result = lj(result, stageCounts, `time)
            result[stageColumn] = long(
                nullFill(result[stageColumn], 0)
            )
            filterNameColumn = symbol([
                "filter" + string(index) + "_name"
            ])[0]
            result[filterNameColumn] = take(
                string(filterColNames[index]),
                result.rows()
            )
        }
    }

    filteredCounts =
        <select count(*) as filtered_count
         from processedTable
         where
            _$timeCol >= startTime,
            _$timeCol < endTime,
            !isNull(_$codeCol)
         group by _$timeCol>.eval()
    filteredCounts.rename!(symbol([timeCol])[0], `time)
    result = lj(result, filteredCounts, `time)
    result[`filtered_count] = long(
        nullFill(result[`filtered_count], 0)
    )
    factorFilterIndex = string(size(filterColNames))
    result["filter" + factorFilterIndex + "_name"] = take(
        "因子空值",
        result.rows()
    )
    result["filter" + factorFilterIndex + "_count"] =
        result[`filtered_count]
    result[`retention_rate] = iif(
        result[`source_count] > 0,
        double(result[`filtered_count]) \
            double(result[`source_count]),
        double(NULL)
    )
    return result
}

def factorInformationCoefficient(
    processedTable,
    returnCols,
    factorCols,
    timeCol="time") {
    returnColNames = factorStringVector(returnCols)
    factorColNames = factorStringVector(factorCols)
    factorCheckColumns(
        processedTable,
        symbol([timeCol])
    )
    factorCheckColumns(processedTable, symbol(factorColNames))
    factorCheckColumns(
        processedTable,
        symbol(returnColNames)
    )

    result =
        <select distinct(_$timeCol) as time
         from processedTable>.eval()
    result = result.sortBy!(`time)

    timeSym = symbol([timeCol])[0]
    for (factorCol in factorColNames) {
        factorSym = symbol([string(factorCol)])[0]
        for (retCol in returnColNames) {
            retSym = symbol([string(retCol)])[0]
            corrTable = sql(
                [
                    sqlCol(timeSym, , `time),
                    sqlColAlias(
                        makeCall(
                            corr,
                            sqlCol(factorSym),
                            sqlCol(retSym)
                        ),
                        `ic
                    ),
                    sqlColAlias(
                        makeCall(
                            spearmanr,
                            sqlCol(factorSym),
                            sqlCol(retSym)
                        ),
                        `rank_ic
                    )
                ],
                processedTable,
                ,
                [sqlCol(timeSym)]
            ).eval()
            result = lj(result, corrTable, `time)
            result.rename!(
                `ic,
                symbol([
                    string(factorCol) + "_" +
                    string(retCol) + "_ic"
                ])[0]
            )
            result.rename!(
                `rank_ic,
                symbol([
                    string(factorCol) + "_" +
                    string(retCol) + "_rank_ic"
                ])[0]
            )
        }
    }
    return result
}

def factorWeightedReturn(weight, ret) {
    if (any(!isNull(weight) && weight < 0)) {
        throw "市值权重不能为负数"
    }
    valid = !isNull(weight) && !isNull(ret)
    validWeight = weight[valid]
    validReturn = ret[valid]
    totalWeight = sum(validWeight)
    if (!isValid(totalWeight) || totalWeight <= 0) {
        return double(NULL)
    }
    return sum(validWeight * validReturn) \ totalWeight
}

def factorExtremeRanks(values, codes, ascending) {
    ranks = take(long(NULL), size(values))
    if (size(values) == 0) return ranks
    ranked = table(
        long(0 .. (size(values) - 1)) as row_id,
        double(values) as factor_value,
        string(codes) as factor_code
    )
    ranked = select * from ranked where !isNull(factor_value)
    ranked.sortBy!(`factor_value`factor_code, [ascending, ascending])
    if (ranked.rows() > 0) {
        ranks[ranked.row_id] = long(0 .. (ranked.rows() - 1))
    }
    return ranks
}

def factorExtremeWeightedReturn(
    factorValue,
    weight,
    ret,
    nSelect,
    ascending,
    codes) {
    selected = !isNull(factorValue) && (
        factorExtremeRanks(factorValue, codes, ascending) < int(nSelect)
    )
    return factorWeightedReturn(weight[selected], ret[selected])
}

def factorValidateGroups(tb, factorCols, nGroups, timeCol="time", codeCol="code") {
    for (factorCol in factorStringVector(factorCols)) {
        groupCol = string(factorCol) + "_group"
        factorCheckColumns(tb, symbol([factorCol, groupCol, timeCol, codeCol]))
        groups = double(tb[groupCol])
        invalid = !isNull(tb[factorCol]) && (
            isNull(groups) || groups < 0 || groups >= nGroups || groups != floor(groups)
        )
        if (any(invalid)) {
            sample = tb[invalid]
            sample = sample[0:min(5, sample.rows())]
            throw groupCol + " 必须是 [0, " + string(nGroups - 1) +
                "] 内的整数；违规行数=" + string(sum(invalid)) +
                "，日期=" + string(sample[timeCol]) + ", code=" + string(sample[codeCol])
        }
    }
}

def factorGroupReturns(
    processedFactorTable,
    returnCols,
    factorCols,
    nGroups,
    nSelect,
    timeCol="time",
    codeCol="code",
    mktmvCol="mktmv") {
    if (nGroups < 2) {
        throw "nGroups must be at least 2"
    }
    if (nSelect < 1) {
        throw "nSelect must be at least 1"
    }
    returnColNames = factorStringVector(returnCols)
    factorColNames = factorStringVector(factorCols)
    factorCheckColumns(
        processedFactorTable,
        symbol([timeCol, codeCol, mktmvCol])
    )
    factorCheckColumns(processedFactorTable, symbol(factorColNames))
    factorCheckColumns(
        processedFactorTable,
        symbol(returnColNames)
    )
    factorValidateGroups(processedFactorTable, factorColNames, nGroups, timeCol, codeCol)

    result =
        <select distinct(_$timeCol) as time
         from processedFactorTable>.eval()
    result = result.sortBy!(`time)

    timeSym = symbol([timeCol])[0]
    mktmvSym = symbol([mktmvCol])[0]
    for (factorCol in factorColNames) {
        factorSym = symbol([string(factorCol)])[0]
        groupCol = string(factorCol) + "_group"
        factorCheckColumns(
            processedFactorTable,
            symbol([groupCol])
        )
        groupSym = symbol([groupCol])[0]
        for (retCol in returnColNames) {
            retSym = symbol([string(retCol)])[0]
            extremes = sql(
                [
                    sqlCol(timeSym, , `time),
                    sqlColAlias(
                        makeCall(
                            factorExtremeWeightedReturn,
                            sqlCol(factorSym),
                            sqlCol(mktmvSym),
                            sqlCol(retSym),
                            nSelect,
                            true,
                            sqlCol(symbol([codeCol])[0])
                        ),
                        `bottom_ret
                    ),
                    sqlColAlias(
                        makeCall(
                            factorExtremeWeightedReturn,
                            sqlCol(factorSym),
                            sqlCol(mktmvSym),
                            sqlCol(retSym),
                            nSelect,
                            false,
                            sqlCol(symbol([codeCol])[0])
                        ),
                        `top_ret
                    )
                ],
                processedFactorTable,
                ,
                [sqlCol(timeSym)]
            ).eval()
            bottomRows =
                <select time, bottom_ret as ret from extremes>.eval()
            result = lj(result, bottomRows, `time)
            result.rename!(
                `ret,
                symbol([
                    string(factorCol) + "_" +
                    string(retCol) + "_bottom"
                ])[0]
            )
            for (groupId in 0 .. (nGroups - 1)) {
                weighted = sql(
                    [
                        sqlCol(timeSym, , `time),
                        sqlColAlias(
                            makeCall(
                                factorWeightedReturn,
                                sqlCol(mktmvSym),
                                sqlCol(retSym)
                            ),
                            `ret
                        )
                    ],
                    processedFactorTable,
                    expr(sqlCol(groupSym), ==, groupId),
                    [sqlCol(timeSym)]
                ).eval()
                result = lj(result, weighted, `time)
                result.rename!(
                    `ret,
                    symbol([
                        string(factorCol) + "_" +
                        string(retCol) + "_group" +
                        string(groupId)
                    ])[0]
                )
            }
            topRows =
                <select time, top_ret as ret from extremes>.eval()
            result = lj(result, topRows, `time)
            result.rename!(
                `ret,
                symbol([
                    string(factorCol) + "_" +
                    string(retCol) + "_top"
                ])[0]
            )
        }
    }
    return result
}

def factorGroupTurnover(
    processedFactorTable,
    factorCols,
    turnoverPeriods,
    nGroups,
    nSelect,
    timeCol="time",
    codeCol="code",
    tradeDates=NULL) {
    if (nGroups < 2) {
        throw "nGroups must be at least 2"
    }
    if (nSelect < 1) {
        throw "nSelect must be at least 1"
    }
    factorColNames = factorStringVector(factorCols)
    periodValues = sort(distinct(int(turnoverPeriods)))
    if (size(periodValues) == 0 || any(periodValues < 1)) {
        throw "turnoverPeriods must contain positive integers"
    }
    factorCheckColumns(
        processedFactorTable,
        symbol([timeCol, codeCol])
    )
    factorCheckColumns(processedFactorTable, symbol(factorColNames))
    factorValidateGroups(processedFactorTable, factorColNames, nGroups, timeCol, codeCol)
    axisDates = tradeDates
    if (typestr(tradeDates) == "VOID") {
        axisDates = <exec _$timeCol from processedFactorTable>.eval()
    }
    factorDates = table(sort(distinct(timestamp(axisDates))) as time)
    factorDates.sortBy!(`time)

    groupColumns = "group" + string(0 .. (nGroups - 1))
    result = table(
        array(TIMESTAMP, 0) as time,
        array(STRING, 0) as factor,
        array(INT, 0) as periods,
        array(DOUBLE, 0) as rank_autocorrelation
    )
    addColumn(result, ["bottom"], take(DOUBLE, 1))
    addColumn(result, groupColumns, take(DOUBLE, nGroups))
    addColumn(result, ["top"], take(DOUBLE, 1))

    for (factorCol in factorColNames) {
        groupCol = string(factorCol) + "_group"
        factorCheckColumns(processedFactorTable, symbol([groupCol]))
        factorMembers = <select
                timestamp(_$timeCol) as time,
                string(_$codeCol) as code,
                double(_$factorCol) as factor_value
            from processedFactorTable
            where !isNull(_$factorCol)>.eval()
        if (factorDates.rows() > 0) {
            factorMembers = select distinct time, code, factor_value
                from factorMembers
            groupMembers = <select
                    timestamp(_$timeCol) as time,
                    string(_$codeCol) as code,
                    int(_$groupCol) as group_id
                from processedFactorTable
                where
                    !isNull(_$factorCol),
                    !isNull(_$groupCol)>.eval()
            groupMembers = select distinct time, code, group_id
                from groupMembers
            ranks = select
                    time,
                    code,
                    rank(
                        factor_value,
                        true,
                        ,
                        true,
                        `average,
                        false
                    ) as factor_rank
                from factorMembers
                context by time
            extremeRanks = select
                    time,
                    code,
                    factorExtremeRanks(
                        factor_value,
                        code,
                        true
                    ) as bottom_rank,
                    factorExtremeRanks(
                        factor_value,
                        code,
                        false
                    ) as top_rank
                from factorMembers
                context by time
            portfolioMembers = select
                    time,
                    code,
                    int(group_id + 1) as portfolio_id
                from groupMembers
            bottomMembers = select
                    time,
                    code,
                    int(0) as portfolio_id
                from extremeRanks
                where bottom_rank < int(nSelect)
            topMembers = select
                    time,
                    code,
                    int(nGroups + 1) as portfolio_id
                from extremeRanks
                where top_rank < int(nSelect)
            portfolioMembers = unionAll(
                portfolioMembers,
                bottomMembers
            )
            portfolioMembers = unionAll(
                portfolioMembers,
                topMembers
            )

            for (period in periodValues) {
                rankDateMap = select
                    time,
                    move(time, period) as previous_time
                    from factorDates
                currentRanks = lj(ranks, rankDateMap, `time)
                previousRanks = select
                        time as previous_time,
                        code,
                        factor_rank as previous_rank
                    from ranks
                rankPairs = lj(currentRanks, previousRanks, ["previous_time", "code"])
                rankRows = select
                        corr(
                            factor_rank,
                            previous_rank
                        ) as rank_autocorrelation
                    from rankPairs
                    where !isNull(previous_rank)
                    group by time

                currentMembers = lj(portfolioMembers, rankDateMap, `time)
                previousMembers = select
                        time as previous_time,
                        code,
                        portfolio_id,
                        true as retained
                    from portfolioMembers
                matchedMembers = lj(currentMembers, previousMembers, ["previous_time", "code", "portfolio_id"])
                turnoverRows = select
                        1.0 - sum(
                            iif(isNull(retained), 0, 1)
                        ) \ count(*) as turnover
                    from matchedMembers
                    where
                        !isNull(previous_time)
                    group by time, portfolio_id

                periodResult = select time from factorDates
                periodResult["factor"] = take(
                    string(factorCol),
                    periodResult.rows()
                )
                periodResult["periods"] = take(
                    int(period),
                    periodResult.rows()
                )
                periodResult = lj(periodResult, rankRows, `time)
                for (portfolioId in 0 .. (nGroups + 1)) {
                    portfolioRows = select time, turnover
                        from turnoverRows
                        where portfolio_id == portfolioId
                    if (portfolioId == 0) {
                        portfolioColumn = `bottom
                    } else if (portfolioId == nGroups + 1) {
                        portfolioColumn = `top
                    } else {
                        portfolioColumn = symbol([
                            "group" + string(portfolioId - 1)
                        ])[0]
                    }
                    portfolioRows.rename!(`turnover, portfolioColumn)
                    periodResult = lj(periodResult, portfolioRows, `time)
                }
                result = unionAll(result, periodResult)
            }
        }
    }
    result.sortBy!(`factor`periods`time)
    return result
}

def analyzeFactor(factorTable, priceTable, settings) {
    print("[DOS 分析] 对齐因子、收益标签和交易日历")
    calendar = exec time from priceTable where code = settings.calendarSymbol
    if (size(calendar) == 0) throw "缺少交易日历基准行情"
    prices = select * from priceTable where time in calendar
    source = factorTable.copy()
    replaceColumn!(prices, `time, timestamp(prices.time))
    replaceColumn!(source, `time, timestamp(source.time))
    replaceColumn!(
        prices,
        `code,
        strReplace(strReplace(string(prices.code), ".SH", ".XSHG"), ".SZ", ".XSHE")
    )
    inputData = lj(source, prices, `time`code)
    if (!settings.marketWeighted) inputData[`circ_mv] = take(1.0, inputData.rows())
    print("[DOS 分析] 保留所有分析因子共同有效的记录")
    processed = factorFilterNulls(inputData, settings.columns)
    if (processed.rows() == 0) throw "没有共同有效的因子样本"
    if (settings.marketWeighted && any(isNull(processed.circ_mv) || processed.circ_mv <= 0))
        throw "市值加权需要每条有效因子记录具备正数 circ_mv"

    for (factorCol in settings.columns) {
        print("[DOS 分组] 开始：" + factorCol)
        groupCount = settings.groups
        ranks = <select int(floor(
            rank(_$factorCol, true, , true, `first, false) * groupCount / double(count(_$factorCol))
        )) as factor_group from processed context by time>.eval()
        processed[factorCol + "_group"] = ranks.factor_group
        print("[DOS 分组] 完成：" + factorCol)
    }
    print("[DOS 覆盖统计] 开始")
    statistics = factorExecutionStatistics(
        source,
        inputData,
        processed,
        array(STRING, 0),
        settings.start,
        settings.end,
        "time",
        "code"
    )
    print("[DOS 覆盖统计] 完成")
    print("[DOS IC] 开始计算 IC 和 RankIC")
    ic = factorInformationCoefficient(processed, settings.returns, settings.columns, "time")
    print("[DOS IC] 完成")
    print("[DOS 分组收益] 开始")
    returns = factorGroupReturns(
        processed,
        settings.returns,
        settings.columns,
        settings.groups,
        settings.selection,
        "time",
        "code",
        "circ_mv"
    )
    print("[DOS 分组收益] 完成")
    if (size(distinct(returns.time)) != returns.rows()) throw "因子统计返回重复交易日期"
    returns = lj(table(calendar as time), returns, `time)
    print("[DOS 换手率] 开始")
    turnover = factorGroupTurnover(
        processed,
        settings.columns,
        settings.periods,
        settings.groups,
        settings.selection,
        "time",
        "code",
        calendar
    )
    print("[DOS 换手率] 完成")
    print("[DOS 分析] 结果保留在服务器，共同有效记录=" + string(processed.rows()))
    return dict(`processed_data`execution_statistics`information_coefficient`group_returns`group_turnover, [
        processed, statistics, ic, returns, turnover
    ])
}
"""


def analyze_factors(
    factor: Factor[P],
    parameters: A,
    *,
    settings: DolphinSettings | None = None,
    checkpoints: Checkpoints | None = None,
) -> FactorAnalysisResult[A]:
    parameters.bind_columns(factor)
    if factor.params.groups != parameters.groups:
        raise ValueError("因子预处理与分析必须使用同一分组数量")
    settings = settings if settings is not None else DolphinSettings.from_env()
    log.info(
        f"[分析] {parameters.start} 至 {parameters.end}，因子={parameters.columns}，"
        f"收益={parameters.return_columns}，分组={parameters.groups}，加权={parameters.weight}"
    )
    panel = checkpoints.load("factor") if checkpoints is not None else None
    restored = panel is not None
    if panel is None:
        panel = factor.compute(parameters.start, parameters.end)
    panel = validate_result(panel, parameters.columns, "因子分析")
    if checkpoints is not None and not restored:
        checkpoints.save("factor", panel)
    source = panel.reset_index().rename(columns={"date": "time", "symbol": "code"})
    source["time"] = pd.to_datetime(source["time"])
    source = source[
        source.time.between(
            pd.Timestamp(parameters.start), pd.Timestamp(parameters.end), inclusive="both"
        )
    ]
    if source.empty:
        raise ValueError("因子面板为空")
    source["code"] = source.code.map(engine_symbol)
    if source.duplicated(["time", "code"]).any():
        raise ValueError("证券代码规范化后存在重复记录")
    for column in parameters.columns:
        values = source[column]
        if (
            pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
            or (
                not pd.api.types.is_numeric_dtype(values)
                and values.map(
                    lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating))
                ).any()
            )
        ):
            raise ValueError("因子值必须是实数")
        source[column] = (
            pd.to_numeric(values, errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)
        )
    source = source.sort_values(["time", "code"])
    codes = sorted(
        {source_symbol(code) for code in source.code} | {source_symbol(parameters.calendar_symbol)}
    )
    derivatives = {
        f"P{parameters.period}L{lag}": {
            "type": "DIRECT",
            "op": "binary.sub",
            "fields": {
                side: {
                    "type": "DIRECT",
                    "op": "unary.log",
                    "fields": {
                        "col": {
                            "type": "TS",
                            "op": "unary.shift",
                            "fields": {"col": parameters.price_column},
                            "params": {"periods": -offset},
                        },
                    },
                }
                for side, offset in (("left", lag + parameters.period), ("right", lag))
            },
        }
        for lag in range(1, parameters.max_lag + 1)
    }
    session = open_session(settings)
    try:
        with query(
            {
                "start_date": parameters.start.isoformat(),
                "end_date": parameters.end.isoformat(),
                "codes": codes,
                "factors": ["circ_mv"] if parameters.weight == "market_value" else [],
                "derivatives": derivatives,
            },
            session=session,
            settings=settings,
        ) as result:
            upload(
                session,
                {
                    "analysisSource": source,
                    "analysisSettings": {
                        "columns": np.asarray(parameters.columns, dtype=str),
                        "returns": np.asarray(parameters.return_columns, dtype=str),
                        "periods": np.asarray([parameters.period], dtype=np.int32),
                        "groups": np.int32(parameters.groups),
                        "selection": np.int32(parameters.n_select),
                        "marketWeighted": parameters.weight == "market_value",
                        "calendarSymbol": source_symbol(parameters.calendar_symbol),
                        "start": np.datetime64(parameters.start, "ms"),
                        "end": np.datetime64(parameters.end, "ms") + np.timedelta64(1, "D"),
                    },
                },
                "因子分析输入",
            )
            session.run(ANALYSIS_SCRIPT)
            session.run(f"""
                {FactorAnalysisResult.data_ref} = analyzeFactor(analysisSource, {result.data_ref}, analysisSettings)
                NULL
            """)
            release(session, ["analysisSource", "analysisSettings"], ["analyzeFactor"])
        report = FactorAnalysisResult(session=session, parameters=parameters.model_copy(deep=True))
    except BaseException as error:
        log.error(f"[分析] 失败：{error_text(error, settings.password.get_secret_value())}")
        try:
            session.close()
        except Exception as close_error:
            log.warning(
                f"[分析] 关闭失败会话异常：{error_text(close_error, settings.password.get_secret_value())}"
            )
        raise
    log.success(f"[分析] 完成：输入 {len(source)} 行，结果保留在服务器，按需读取")
    return report
