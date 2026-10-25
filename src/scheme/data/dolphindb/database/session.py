import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import dolphindb as ddb
from runtime.database.session import redirect_session_output
from runtime.utils import logger as log

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
    """仅在建立连接时依次尝试端点，不重连或重放任务，不设置查询超时。"""
    failures = []
    for endpoint in settings.endpoints:
        host, port = endpoint.rsplit(":", 1)
        host = host.removeprefix("[").removesuffix("]")
        session = ddb.session()
        connected = False
        try:
            redirect_session_output(session)
            connected = session.connect(
                host,
                int(port),
                settings.username,
                settings.password.get_secret_value(),
                highAvailability=False,
                reconnect=False,
            )
            if not connected:
                raise ConnectionError("服务端未接受连接")
        except Exception as error:
            message = error_text(error, settings.password.get_secret_value())
            failures.append(f"{endpoint}：{message}")
            log.warning(f"[连接] DolphinDB {endpoint} 连接失败：{message}")
        finally:
            if not connected:
                try:
                    session.close()
                except Exception as error:
                    message = error_text(error, settings.password.get_secret_value())
                    log.warning(f"[连接] DolphinDB {endpoint} 失败会话关闭异常：{message}")
        if connected:
            return session
    raise ConnectionError(f"所有 DolphinDB 端点均连接失败：{'；'.join(failures)}") from None


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
