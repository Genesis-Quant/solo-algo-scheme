from typing import TypeVar

import numpy as np
import pandas as pd
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.base import Factor, FactorAnalysisParams, FactorParams
from scheme.base.projects.factor.algo import validate_result
from scheme.config import DolphinSettings
from scheme.data import query
from scheme.data.dolphindb.database.session import open_session, release, upload
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol

from .result import FactorAnalysisResult

P = TypeVar("P", bound=FactorParams)
A = TypeVar("A", bound=FactorAnalysisParams)

ANALYSIS_SCRIPT = """
use factor

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
    processed = factor::factorFilterNulls(inputData, settings.columns)
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
    statistics = factor::factorExecutionStatistics(
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
    ic = factor::factorInformationCoefficient(processed, settings.returns, settings.columns, "time")
    print("[DOS IC] 完成")
    print("[DOS 分组收益] 开始")
    returns = factor::factorGroupReturns(
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
    turnover = factor::factorGroupTurnover(
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
) -> FactorAnalysisResult[A]:
    parameters.bind_columns(factor)
    if factor.params.groups != parameters.groups:
        raise ValueError("因子预处理与分析必须使用同一分组数量")
    settings = settings if settings is not None else DolphinSettings.from_env()
    log.info(
        f"[分析] {parameters.start} 至 {parameters.end}，因子={parameters.columns}，"
        f"收益={parameters.return_columns}，分组={parameters.groups}，加权={parameters.weight}"
    )
    panel = validate_result(
        factor.compute(parameters.start, parameters.end), parameters.columns, "因子分析"
    )
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
