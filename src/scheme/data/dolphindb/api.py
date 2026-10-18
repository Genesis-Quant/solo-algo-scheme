"""首次空 codes 查询时初始化进程级全时段证券域，结果支持 with。

模块导入不连接数据库。每个研究进程第一次空 codes 查询读取 CoreData 全时间范围内
所有出现过记录的代码，不按输出日期或 close 筛选；之后空 codes 查询复用该快照，
显式 codes 原样保留。日期轴和成员填充沿用 Runtime 规则。
"""

import json
from datetime import timedelta
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from runtime.apps.query import FactorQuery
from runtime.apps.query import api as _runtime
from runtime.apps.query.result import QueryResult

from .backtest import backtest

__all__ = ["query", "backtest"]

_ALL_CODES: tuple[str, ...] | None = None
_CODES_LOCK = Lock()


def query(
    request: FactorQuery | dict[str, Any],
    *,
    session: Any | None = None,
) -> QueryResult:
    """补齐空 codes 后执行单次查询；成功后会话交由 Runtime QueryResult 持有。"""
    global _ALL_CODES
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
    owns_session = session is None
    if owns_session:
        session = _runtime.create_session(max_time=_runtime.QUERY_SESSION_MAX_TIME)
    try:
        _runtime.redirect_session_output(session)
        if not validated.codes:
            with _CODES_LOCK:
                if _ALL_CODES is None:
                    _ALL_CODES = tuple(str(code) for code in session.run(
                        "exec string(code) from (select count(*) as records from "
                        f"{_runtime.CORE_TABLE} group by code) "
                        "where not isNull(code) order by code"
                    ))
            validated = validated.model_copy(update={"codes": list(_ALL_CODES)})

        # 代码轴已固定，只读取真实交易日，不再查询当前区间的 close 代码并集。
        session.upload({"coreAxisStart": calculation_start, "coreAxisEnd": output_end + timedelta(days=1)})
        dates = pd.DatetimeIndex(session.run(f"""
            exec distinct date(time) from {_runtime.CORE_TABLE}
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
        session.run("use query")
        if not _runtime.has_session_variable(session, _runtime.SOURCE_REF):
            session.run(f"""
                {_runtime.SOURCE_REF} = build_factor_source(
                    {_runtime.CORE_TABLE}, coreQueryCodes, coreQueryFactors,
                    coreQueryDates, coreQueryStart, coreQueryEnd, coreQuerySeedFactors
                )
            """)
            for factor in source_factors:
                name = json.dumps(factor, ensure_ascii=False)
                if factor == _runtime.IS_ST_FACTOR:
                    session.run(f"""
                        {_runtime.SOURCE_REF} = fill_null_column({_runtime.SOURCE_REF}, {name}, 0.0)
                    """)
                elif factor.startswith(_runtime.WEIGHT_PREFIX):
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
                    session.run(f"""
                        {_runtime.SOURCE_REF} = forward_fill_column(
                            {_runtime.SOURCE_REF}, {name}, {_runtime.SOURCE_REF}.code,
                            {_runtime.SOURCE_REF}.time
                        )
                    """)
            session.run(f"""
                {_runtime.SOURCE_REF} = finalize_factor_source({_runtime.SOURCE_REF}, coreQueryFactors)
            """)
        session.run(f"""
            {_runtime.COMPUTED_REF} = compute_factors(
                {_runtime.SOURCE_REF}, fromStdJson(coreDslDefinitionsJson)
            )
            {_runtime.FILTERED_REF} = filter_factors({_runtime.COMPUTED_REF}, coreDslFilters)
            {_runtime.DATA_REF} = project_factor_output(
                {_runtime.FILTERED_REF}, coreDslOutputColumns, coreOutputStart, coreOutputEnd
            )
        """)
        return QueryResult(
            session=session, source_ref=_runtime.SOURCE_REF, computed_ref=_runtime.COMPUTED_REF,
            filtered_ref=_runtime.FILTERED_REF, data_ref=_runtime.DATA_REF,
        )
    except BaseException:
        if owns_session:
            session.close()
        raise
