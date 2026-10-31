"""独立 Query 包的 Solo 兼容入口。"""

from typing import Any

from query import FactorQuery
from query.apps.query import api as query_api
from query.apps.query.result import QueryResult
from query.database.compile.query.scripts import QUERY_SCRIPT

from scheme.config import DolphinSettings

__all__ = ["FactorQuery", "QUERY_SCRIPT", "QueryResult", "query"]


def query(
    request: FactorQuery | dict[str, Any],
    *,
    session: Any | None = None,
    settings: DolphinSettings | None = None,
) -> QueryResult:
    return query_api.execute_query(request, session=session, settings=settings)
