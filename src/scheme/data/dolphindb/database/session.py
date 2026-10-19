from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import dolphindb as ddb
from runtime.database.session import redirect_session_output
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.config import DolphinSettings


def open_session(settings: DolphinSettings) -> Any:
    """连接显式端点，不设置读写超时或会话寿命；返回后由调用方负责关闭。"""
    log.info("DolphinDB: {}:{}，{}/{}",
             settings.host, settings.port, settings.database, settings.table)
    session = ddb.session()
    try:
        redirect_session_output(session)
        if not session.connect(
            settings.host, settings.port, settings.username, settings.password.get_secret_value(),
        ):
            raise ConnectionError(f"无法连接 DolphinDB {settings.host}:{settings.port}")
        log.debug("DolphinDB 已连接")
        return session
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
