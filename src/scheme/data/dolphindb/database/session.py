from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import dolphindb as ddb
from runtime.database.session import DeadlineSession, redirect_session_output
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.config import DolphinSettings


def open_session(settings: DolphinSettings, *, max_time: int | None = None) -> Any:
    """连接显式端点；返回后由调用方负责关闭。"""
    if max_time is not None and (
        isinstance(max_time, bool) or not isinstance(max_time, int) or max_time <= 0
    ):
        raise ValueError("max_time 必须是正整数秒数或 None")
    log.info("DolphinDB: {}:{}，{}/{}",
             settings.host, settings.port, settings.database, settings.table)
    session = ddb.session()
    try:
        redirect_session_output(session)
        timeouts = {} if max_time is None else {"readTimeout": max_time, "writeTimeout": max_time}
        if not session.connect(
            settings.host, settings.port, settings.username, settings.password.get_secret_value(),
            **timeouts,
        ):
            raise ConnectionError(f"无法连接 DolphinDB {settings.host}:{settings.port}")
        log.debug("DolphinDB 已连接，读写及会话总限时={}s", max_time)
        return session if max_time is None else DeadlineSession(session, max_time)
    except BaseException as error:
        log.error("DolphinDB 连接失败：{}", error_text(error, settings.password.get_secret_value()))
        session.close()
        raise


@contextmanager
def create_session(settings: DolphinSettings) -> Iterator[Any]:
    session = open_session(settings)
    try:
        yield session
    finally:
        session.close()
        log.debug("DolphinDB 连接已关闭：{}:{}", settings.host, settings.port)
