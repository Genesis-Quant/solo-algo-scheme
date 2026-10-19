"""空 codes 使用 Arena Runtime 缓存的 Tushare stock_basic 股票列表。"""

import json
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from runtime.apps.query import FactorQuery
from runtime.apps.query import api as _runtime
from runtime.apps.query.result import QueryResult
from runtime.utils import get_codes
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.config import DolphinSettings

from .database.session import open_session

__all__ = ["query"]


def query(
    request: FactorQuery | dict[str, Any],
    *,
    session: Any | None = None,
    settings: DolphinSettings | None = None,
) -> QueryResult:
    """使用 SOLO 数据连接查询；成功后会话交由 Runtime QueryResult 持有。"""
    validated = _runtime.validate_factor_query(request)
    output_start, output_end = _runtime.normalize_date_range(
        validated.start_date, validated.end_date,
    )
    calculation_start = (output_start - validated.lookback).normalize()
    source_factors = validated.source_factors()
    if unknown := set(source_factors) - set(_runtime.available_factors()):
        raise ValueError(f"查询包含 Worker 未声明的字段：{sorted(unknown)}")
    output_columns = [
        _runtime.TIME_COLUMN, _runtime.CODE_COLUMN, *validated.factors, *validated.derivatives,
    ]
    seed_factors = [factor for factor in source_factors if factor in _runtime.INDUSTRY_FACTORS]
    settings = settings if settings is not None else DolphinSettings.from_env()
    core_table = (
        f"loadTable({json.dumps(settings.database, ensure_ascii=False)}, "
        f"{json.dumps(settings.table, ensure_ascii=False)})"
    )
    owns_session = session is None
    log.info(
        "查询 {} 至 {}（含），股票={}，字段={}，衍生列={}，筛选={}；数据源={}:{}",
        output_start.date(), output_end.date(),
        len(validated.codes) if validated.codes else "stock_basic 全部股票",
        validated.factors, list(validated.derivatives), validated.filters,
        settings.host, settings.port,
    )
    try:
        if owns_session:
            session = open_session(settings)
        else:
            _runtime.redirect_session_output(session)
            log.debug("查询复用调用方连接：{}:{}，{}/{}",
                      settings.host, settings.port, settings.database, settings.table)
        if not validated.codes:
            codes = get_codes()
            validated = validated.model_copy(update={"codes": list(codes)})
            log.debug("stock_basic 股票域：{} 只，复用 Runtime 进程缓存", len(codes))

        log.debug("交易日历范围 {} 至 {}，包含 lookback {}",
                  calculation_start.date(), output_end.date(), validated.lookback)
        session.upload({"coreAxisStart": calculation_start, "coreAxisEnd": output_end + timedelta(days=1)})
        dates = pd.DatetimeIndex(session.run(f"""
            exec distinct date(time) from {core_table}
            where factor = `close, time >= coreAxisStart, time < coreAxisEnd
            order by date(time)
        """))
        if dates.empty:
            raise ValueError(
                f"业务因子表在 {calculation_start:%Y-%m-%d} 至 {output_end:%Y-%m-%d} 没有收盘价数据"
            )
        definitions = {
            name: node.model_dump(mode="json") for name, node in validated.derivatives.items()
        }
        session.upload({
            "coreQueryStart": calculation_start,
            "coreQueryEnd": output_end + timedelta(days=1),
            "coreQueryCodes": np.asarray(validated.codes, dtype=str),
            "coreQueryFactors": np.asarray(source_factors, dtype=str),
            "coreQuerySeedFactors": np.asarray(seed_factors, dtype=str),
            "coreQueryDates": dates.to_numpy(dtype="datetime64[ms]"),
            "coreDslDefinitionsJson": json.dumps(definitions, ensure_ascii=False, separators=(",", ":")),
            "coreDslFilters": np.asarray(validated.filters, dtype=str),
            "coreDslOutputColumns": np.asarray(output_columns, dtype=str),
            "coreOutputStart": output_start,
            "coreOutputEnd": output_end + timedelta(days=1),
        })
        log.debug("使用 query 模块；基础字段={}，历史种子字段={}", source_factors, seed_factors)
        session.run("use query")
        source_cached = _runtime.has_session_variable(session, _runtime.SOURCE_REF, log_progress=False)
        log.debug("基础数据缓存={}；股票={}，历史交易日={}", source_cached, len(validated.codes), len(dates))
        if not source_cached:
            session.run(f"""
                {_runtime.SOURCE_REF} = build_factor_source(
                    {core_table}, coreQueryCodes, coreQueryFactors,
                    coreQueryDates, coreQueryStart, coreQueryEnd, coreQuerySeedFactors
                )
            """)
            for factor in source_factors:
                name = json.dumps(factor, ensure_ascii=False)
                if factor == _runtime.IS_ST_FACTOR:
                    log.debug("字段 {}：缺失填零", factor)
                    session.run(f"""
                        {_runtime.SOURCE_REF} = fill_null_column({_runtime.SOURCE_REF}, {name}, 0.0)
                    """)
                elif factor.startswith(_runtime.WEIGHT_PREFIX):
                    log.debug("字段 {}：观测日补零后按证券前向填充", factor)
                    session.run(f"""
                        {_runtime.SOURCE_REF} = fill_observed_group_null_column(
                            {_runtime.SOURCE_REF}, {name}, {_runtime.SOURCE_REF}.time, 0.0
                        )
                        {_runtime.SOURCE_REF} = forward_fill_column(
                            {_runtime.SOURCE_REF}, {name}, {_runtime.SOURCE_REF}.code,
                            {_runtime.SOURCE_REF}.time
                        )
                    """)
                elif factor in _runtime.FINANCIAL_FACTORS or factor in _runtime.INDUSTRY_FACTORS:
                    log.debug("字段 {}：按证券前向填充", factor)
                    session.run(f"""
                        {_runtime.SOURCE_REF} = forward_fill_column(
                            {_runtime.SOURCE_REF}, {name}, {_runtime.SOURCE_REF}.code,
                            {_runtime.SOURCE_REF}.time
                        )
                    """)
            session.run(f"""
                {_runtime.SOURCE_REF} = finalize_factor_source({_runtime.SOURCE_REF}, coreQueryFactors)
            """)
        log.debug("计算 DSL：{}", list(validated.derivatives))
        session.run(f"""
            {_runtime.COMPUTED_REF} = compute_factors(
                {_runtime.SOURCE_REF}, fromStdJson(coreDslDefinitionsJson)
            )
        """)
        session.run(f"""
            {_runtime.FILTERED_REF} = filter_factors({_runtime.COMPUTED_REF}, coreDslFilters)
        """)
        row_count = session.run(f"""
            {_runtime.DATA_REF} = project_factor_output(
                {_runtime.FILTERED_REF}, coreDslOutputColumns, coreOutputStart, coreOutputEnd
            )
            {_runtime.DATA_REF}.rows()
        """)
        result = QueryResult(
            session=session, source_ref=_runtime.SOURCE_REF, computed_ref=_runtime.COMPUTED_REF,
            filtered_ref=_runtime.FILTERED_REF, data_ref=_runtime.DATA_REF,
        )
        log.info(
            "查询返回 {} 行、{} 列，股票域 {} 只，范围 {} 至 {}",
            row_count, len(output_columns), len(validated.codes),
            output_start.date(), output_end.date(),
        )
        if row_count == 0:
            log.warning("查询结果为空：{} 至 {}，筛选={}",
                        output_start.date(), output_end.date(), validated.filters)
        return result
    except BaseException as error:
        log.error(
            "查询失败：{}:{}，范围 {} 至 {}；{}",
            settings.host, settings.port, output_start.date(), output_end.date(),
            error_text(error, settings.password.get_secret_value()),
        )
        if owns_session and session is not None:
            try:
                session.close()
            except Exception as close_error:
                log.warning("查询失败后的连接回收失败：{}",
                            error_text(close_error, settings.password.get_secret_value()))
        raise
