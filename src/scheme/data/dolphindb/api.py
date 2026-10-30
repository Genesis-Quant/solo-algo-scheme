"""在 DolphinDB 执行查询，Python 只传入条件并按需取得最终结果。"""

import json
from datetime import timedelta
from typing import Any

import numpy as np
from runtime.apps.query import FactorQuery as RuntimeFactorQuery
from runtime.apps.query import api as query_api
from runtime.utils import get_codes
from runtime.utils import logger as log

from scheme.config import DolphinSettings
from scheme.errors import error_text

from .database.session import download, open_session, release, upload
from .schema import FactorQuery

__all__ = ["query"]

QUERY_SCRIPT = """
use query

def buildQuery(request) {
    sourceTable = loadTable(request.database, request.table)
    calendar = exec distinct timestamp(date(time)) from sourceTable
        where factor = `close, time >= request.start, time < request.end
        order by timestamp(date(time))
    if (size(calendar) == 0) throw "查询区间没有交易日历"

    timeline = calendar
    for (column in request.weights) {
        observedDates = exec distinct timestamp(time) from sourceTable
            where factor = symbol([column])[0], time >= request.start, time < request.end
        timeline = sort(distinct(timeline join observedDates))
    }
    source = build_factor_source(
        sourceTable,
        request.codes,
        request.factors,
        timeline,
        request.start,
        request.end,
        request.industry
    )
    source[`selected] = source.time in calendar

    for (column in request.factors) {
        if (column == request.st) {
            source = fill_null_column(source, column, 0.0)
        } else if (column in request.weights) {
            observedDates = exec distinct timestamp(time) from sourceTable
                where factor = symbol([column])[0], time >= request.start, time < request.end
            source[column] = iif(
                source.time in observedDates,
                nullFill(source[column], 0.0),
                source[column]
            )
            source = forward_fill_column(source, column, source.code, source.time)
            snapshotDate = exec max(time) from sourceTable
                where factor = symbol([column])[0], time < request.start
            if (!isNull(snapshotDate)) {
                snapshot = select first(value) as weight from sourceTable
                    where factor = symbol([column])[0], time = snapshotDate
                    group by code
                prior = dict(string(snapshot.code), double(snapshot.weight))[string(source.code)]
                source[column] = iif(isNull(source[column]), nullFill(prior, 0.0), source[column])
            }
        } else if (column in request.financial || column in request.industry) {
            source = forward_fill_column(source, column, source.code, source.time)
        }
    }
    source = finalize_factor_source(source, request.factors)
    computed = compute_factors(source, fromStdJson(request.definitions))
    filtered = filter_factors(computed, request.filters)
    data = project_factor_output(filtered, request.columns, request.outputStart, request.outputEnd)
    calendar = calendar[calendar >= request.outputStart && calendar < request.outputEnd]
    return dict(`data`calendar, [data, calendar])
}
"""


class QueryResult:
    """创建并持有当前查询的数据和日历；关闭结果时只释放自己的资源。"""

    data_ref = "queryData"
    calendar_ref = "dates"

    def __init__(
            self,
            parameters: dict[str, Any],
            *,
            session: Any | None,
            settings: DolphinSettings,
    ) -> None:
        self.owns_session = session is None
        self.closed = False
        self.session = open_session(settings) if self.owns_session else session
        started = False
        try:
            query_api.redirect_session_output(self.session)
            self.session.run(f"""
                if ({json.dumps(self.data_ref)} in objs(true).name || {json.dumps(self.calendar_ref)} in objs(true).name)
                    throw "当前连接仍有未关闭的查询结果，请先关闭再查询"
                NULL
            """)
            started = True
            try:
                upload(self.session, {"queryRequest": parameters}, "查询条件")
                self.session.run(QUERY_SCRIPT)
                self.session.run(f"""
                    queryResult = buildQuery(queryRequest)
                    {self.data_ref} = queryResult.data
                    {self.calendar_ref} = queryResult.calendar
                    NULL
                """)
            finally:
                release(self.session, ["queryRequest", "queryResult"], ["buildQuery"])
        except BaseException:
            if self.owns_session or started:
                try:
                    self.close()
                except Exception as error:
                    log.warning(
                        f"[查询] 查询失败后关闭结果异常：{error_text(error, settings.password.get_secret_value())}"
                    )
            raise

    @property
    def data(self) -> Any:
        if self.closed:
            raise RuntimeError("查询结果已关闭")
        return download(self.session, self.data_ref, "查询结果")

    @property
    def calendar(self) -> Any:
        if self.closed:
            raise RuntimeError("查询结果已关闭")
        return download(self.session, self.calendar_ref, "交易日历")

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.owns_session:
                self.session.close()
            else:
                release(self.session, [self.data_ref, self.calendar_ref])
        finally:
            self.closed = True

    def __enter__(self) -> "QueryResult":
        if self.closed:
            raise RuntimeError("查询结果已关闭")
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        try:
            self.close()
        except Exception as error:
            if exception_type is None:
                raise
            log.warning(f"[查询] 异常退出时关闭结果失败：{error_text(error)}")
        return False


def query(
        request: RuntimeFactorQuery | dict[str, Any],
        *,
        session: Any | None = None,
        settings: DolphinSettings | None = None,
) -> QueryResult:
    settings = settings if settings is not None else DolphinSettings.from_env()
    if isinstance(request, RuntimeFactorQuery):
        validated = (
            request
            if isinstance(request, FactorQuery)
            else FactorQuery.model_validate(request.model_dump())
        )
    elif isinstance(request, dict):
        validated = FactorQuery.model_validate(request)
    else:
        raise TypeError("request 必须是 FactorQuery 或 dict[str, Any]")
    start, end = query_api.normalize_date_range(validated.start_date, validated.end_date)
    factors = validated.source_factors()
    if unknown := set(factors) - set(query_api.available_factors()):
        raise ValueError(f"查询包含未声明的字段：{sorted(unknown)}")
    codes = validated.codes or list(get_codes())
    if not codes:
        raise ValueError("查询证券列表为空")
    parameters = {
        "database": settings.database,
        "table": settings.table,
        "start": (start - validated.lookback).normalize(),
        "end": (end + validated.lookforward).normalize() + timedelta(days=1),
        "outputStart": start,
        "outputEnd": end + timedelta(days=1),
        "codes": np.asarray(codes, dtype=str),
        "factors": np.asarray(factors, dtype=str),
        "industry": np.asarray([f for f in factors if f in query_api.INDUSTRY_FACTORS], dtype=str),
        "financial": np.asarray([f for f in factors if f in query_api.FINANCIAL_FACTORS], dtype=str),
        "weights": np.asarray([f for f in factors if f.startswith(query_api.WEIGHT_PREFIX)], dtype=str),
        "st": query_api.IS_ST_FACTOR,
        "definitions": json.dumps(
            {name: node.model_dump(mode="json") for name, node in validated.derivatives.items()},
            ensure_ascii=False, separators=(",", ":"),
        ),
        "filters": np.asarray(validated.filters, dtype=str),
        "columns": np.asarray(["time", "code", *validated.factors, *validated.derivatives], dtype=str),
    }

    try:
        return QueryResult(parameters, session=session, settings=settings)
    except BaseException as error:
        log.error(f"[查询] 失败：{error_text(error, settings.password.get_secret_value())}")
        raise
