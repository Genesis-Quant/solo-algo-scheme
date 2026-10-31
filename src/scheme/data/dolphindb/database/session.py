import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from query.database.session import create_session as open_query_session
from query.database.session import redirect_session_output
from query.utils import logger as log

from scheme.config import DolphinSettings
from scheme.errors import error_text


def upload(session: Any, variables: dict[str, Any], label: str) -> None:
    try:
        redirect_session_output(session)
        session.upload(variables)
    except BaseException:
        log.error(f"[上传] {label}失败")
        raise


def release(
    session: Any,
    variables: Sequence[str] = (),
    functions: Sequence[str] = (),
) -> None:
    statements = [
        f"if ({json.dumps(name)} in objs(true).name) undef({json.dumps(name)}, VAR)"
        for name in variables
    ]
    statements.extend(
        f"if ({json.dumps(name)} in defs().name) undef({json.dumps(name)}, DEF)"
        for name in functions
    )
    if not statements:
        return
    try:
        session.run("\n".join([*statements, "NULL"]))
    except Exception as error:
        log.warning(f"[清理] 临时数据回收失败：{error_text(error)}")


def download(session: Any, script: str, label: str) -> Any:
    if not isinstance(script, str) or not script.strip():
        raise ValueError("读取表达式不能为空")
    try:
        redirect_session_output(session)
        return session.run(script)
    except BaseException:
        log.error(f"[读取] {label}失败")
        raise


def open_session(settings: DolphinSettings) -> Any:
    """连接后预加载当前 Query 包编译的 DOS，不重连或重放任务。"""
    return open_query_session(settings)


@contextmanager
def create_session(settings: DolphinSettings) -> Iterator[Any]:
    session = open_session(settings)
    try:
        yield session
    finally:
        failed = sys.exception() is not None
        identifier = session.session_id
        try:
            session.close()
        except Exception as error:
            if not failed:
                raise
            log.warning(
                f"[连接] 异常退出时关闭会话失败：{error_text(error, settings.password.get_secret_value())}"
            )
        else:
            log.info(f"[连接] DolphinDB 会话已关闭：{identifier}")
