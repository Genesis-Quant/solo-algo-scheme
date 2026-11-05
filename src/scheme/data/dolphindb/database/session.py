"""DolphinDB会话创建、生命周期及数据交换，不初始化Query脚本。"""

import json
from collections.abc import Sequence
from threading import Lock, Timer
from time import monotonic
from typing import Any, Literal, Self

import dolphindb

from scheme import logger
from scheme.data.dolphindb.settings import DolphinSettings
from scheme.errors import error_text


class LoguruSink(dolphindb.logger.Sink):
    def __init__(self) -> None:
        super().__init__("loguru")

    def handle(self, message: Any) -> None:
        logger.log(message.level.name, message.log.rstrip("\r\n"))

    def flush(self) -> None:
        return None


def redirect_session_output(session: Any) -> None:
    message_logger = session.msg_logger
    message_logger.disable_stdout_sink()
    if not any(sink.name == "loguru" for sink in message_logger.list_sinks()):
        message_logger.add_sink(LoguruSink())


class Session:
    """持有SDK连接，支持with、幂等close及可选会话总寿命。"""

    def __init__(self, session: Any, max_time: int | None = None) -> None:
        self.raw_session = session
        self.max_time = max_time
        self.deadline = monotonic() + max_time if max_time is not None else None
        self.lock = Lock()
        self.closed = False
        self.expired = False
        self.timer = None
        if max_time is not None:
            self.timer = Timer(max_time, self.expire)
            self.timer.daemon = True
            self.timer.start()

    def ensure_open(self) -> None:
        if self.expired or self.deadline is not None and monotonic() >= self.deadline:
            self.expire()
            raise TimeoutError(f"DolphinDB Session 已超过最长使用时间 {self.max_time} 秒")
        if self.closed:
            raise RuntimeError("DolphinDB Session 已关闭")

    def run(self, script: str, *args: Any, **kwargs: Any) -> Any:
        self.ensure_open()
        try:
            result = self.raw_session.run(script, *args, **kwargs)
        except Exception as error:
            if self.expired:
                raise TimeoutError("DolphinDB Session 已超过最长使用时间") from error
            raise
        self.ensure_open()
        return result

    def upload(self, variables: dict[str, Any]) -> Any:
        self.ensure_open()
        result = self.raw_session.upload(variables)
        self.ensure_open()
        return result

    def expire(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.expired = True
            self.closed = True
        try:
            self.raw_session.close()
        except Exception as error:
            logger.warning(f"DolphinDB Session 超时关闭失败：{type(error).__name__}")

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.timer is not None:
                self.timer.cancel()
        self.raw_session.close()

    def __enter__(self) -> Self:
        self.ensure_open()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        try:
            self.close()
        except Exception as error:
            if exception_type is None:
                raise
            logger.warning(f"异常退出时关闭DolphinDB会话失败：{type(error).__name__}")
        return False

    @property
    def msg_logger(self) -> Any:
        self.ensure_open()
        return self.raw_session.msg_logger

    @property
    def session_id(self) -> Any:
        self.ensure_open()
        return self.raw_session.session_id


def create_session(
        settings: type[DolphinSettings] = DolphinSettings,
        *,
        role: Literal["runtime", "worker"] = "runtime",
        redirect_output: bool = True,
        max_time: int | None = None,
) -> Session:
    """创建普通SDK会话；仅建连时尝试端点，不重连、重放或加载查询脚本。"""
    if max_time is not None and (
        isinstance(max_time, bool) or not isinstance(max_time, int) or max_time <= 0
    ):
        raise ValueError("max_time 必须是正整数秒数或 None")
    if role == "runtime":
        username, password = settings.username, settings.password
    elif role == "worker":
        username, password = settings.worker_username, settings.worker_password
    else:
        raise ValueError(f"未知 DolphinDB 账号角色：{role}")
    if not username or not password:
        raise RuntimeError(f"缺少 DolphinDB {role} 专用账号配置")
    failures = []
    for endpoint in settings.endpoints:
        host, port = endpoint.rsplit(":", 1)
        session = dolphindb.session()
        try:
            if redirect_output:
                redirect_session_output(session)
            if not session.connect(
                host.strip("[]"), int(port), username, password,
                highAvailability=False, reconnect=False,
            ):
                raise ConnectionError("服务端未接受连接")
        except BaseException as error:
            try:
                session.close()
            except Exception as close_error:
                logger.warning(f"连接失败后关闭会话异常：{type(close_error).__name__}")
            if not isinstance(error, Exception):
                raise
            failures.append(f"{endpoint}：{error_text(error, password)}")
        else:
            try:
                return Session(session, max_time)
            except BaseException:
                try:
                    session.close()
                except Exception as close_error:
                    logger.warning(f"初始化失败后关闭会话异常：{type(close_error).__name__}")
                raise
    raise ConnectionError(f"所有 DolphinDB 端点均连接失败：{'；'.join(failures)}") from None


def upload(session: Any, variables: dict[str, Any], label: str) -> None:
    try:
        redirect_session_output(session)
        session.upload(variables)
    except BaseException:
        logger.error(f"[上传] {label}失败")
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
        logger.warning(f"[清理] 临时数据回收失败：{error_text(error)}")


def download(session: Any, script: str, label: str) -> Any:
    if not isinstance(script, str) or not script.strip():
        raise ValueError("读取表达式不能为空")
    try:
        redirect_session_output(session)
        return session.run(script)
    except BaseException:
        logger.error(f"[读取] {label}失败")
        raise


__all__ = ["Session", "create_session", "download", "release", "upload"]
