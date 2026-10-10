"""沿用 Arena 日志格式，为研究进程设置级别并接入标准库日志。"""

import logging
import os
import re
import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TextIO

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger, Record

__all__ = ["configure", "get_logger", "is_enabled"]

LEVEL_VARIABLE = "SCHEME_LOG_LEVEL"
_DETAIL_SOURCES = ("runtime.apps.query", "runtime.database.session")
_RESEARCH_SOURCES = re.compile(
    r"(?:backtest|scheme|__main__"
    r"|(?:factor|model|optimize|control|execution|strategy)(?:_[0-9a-f]{4})?)(?:\.|$)"
)
_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}
_simulation: ContextVar[str] = ContextVar("scheme_log_simulation", default="")
_current: tuple[str, TextIO] | None = None
_threshold = logger.level("INFO").no
_intercept: logging.Handler | None = None


class _Intercept(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING and not _RESEARCH_SOURCES.match(record.name):
            return
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        try:
            message = record.getMessage()
        except Exception:
            self.handleError(record)
            return

        def source(entry: "Record") -> None:
            # 标准库记录可能经过队列或 Handler 转发，不能用桥接函数的调用栈冒充来源。
            entry["file"] = type(entry["file"])(os.path.basename(record.pathname), record.pathname)
            entry["function"] = record.funcName or "-"
            entry["line"] = record.lineno
            entry["module"] = record.module
            entry["name"] = record.name

        logger.patch(source).opt(exception=record.exc_info).log(level, message)


def _patch(record: "Record") -> None:
    moment = _simulation.get()
    if moment:
        record["message"] = f"[回测 {moment}] {record['message']}"


def _accept(threshold: int):
    debug = logger.level("DEBUG").no
    warning = logger.level("WARNING").no

    def accept(record: "Record") -> bool:
        if record["level"].no < warning and (record["name"] or "").startswith(_DETAIL_SOURCES):
            return threshold <= debug
        return True

    return accept


def is_enabled(level: str) -> bool:
    """按当前 Scheme 阈值判断级别，避免为不会输出的日志准备数据。"""
    name = level.strip().upper()
    return logger.level(_ALIASES.get(name, name)).no >= _threshold


def configure(level: str | None = None, *, stream: TextIO | None = None) -> None:
    """沿用 Arena 的文件、函数和行号格式；默认 INFO，可通过 SCHEME_LOG_LEVEL 调整。"""
    # Arena 导入时设置 loguru，必须先完成导入，再接管级别和输出流。
    from runtime.utils.logging import log_format

    invalid = None
    if level is not None:
        name = _ALIASES.get(level.strip().upper(), level.strip().upper())
        threshold = logger.level(name).no
    else:
        name = (os.environ.get(LEVEL_VARIABLE) or "INFO").strip().upper()
        name = _ALIASES.get(name, name)
        try:
            threshold = logger.level(name).no
        except ValueError:
            invalid, name = name, "INFO"
            threshold = logger.level(name).no
    # stdout 留给 CLI JSON；Notebook 写 stdout，避免正常研究日志被显示为错误流。
    in_kernel = type(sys.stdout).__module__.startswith("ipykernel")
    target = stream or (sys.stdout if in_kernel else sys.stderr)
    global _current, _threshold, _intercept
    _threshold = threshold
    if _current != (name, target):
        logger.remove()
        logger.configure(patcher=_patch)
        logger.add(
            target, level=threshold, format=log_format, filter=_accept(threshold),
            colorize=None, backtrace=False, diagnose=False,
        )
        _current = (name, target)
    standard = logging.getLevelName(name)
    standard = standard if isinstance(standard, int) else threshold
    root = logging.getLogger()
    # 宿主负责已有及后加的 handlers；只安装空 root，或调整仍由 Scheme 独占的级别。
    if not root.handlers:
        _intercept = _Intercept()
        logging.basicConfig(handlers=[_intercept], level=standard)
    elif _intercept is not None and root.handlers == [_intercept]:
        root.setLevel(standard)
    if invalid:
        get_logger("scheme").warning("{} 无效：{}，已按 INFO 输出日志", LEVEL_VARIABLE, invalid)


def get_logger(component: str) -> "Logger":
    """返回研究日志实例；文本来源仍使用调用方的文件、函数和行号。"""
    return logger.bind(component=component)


def mark_simulation(moment: Any | None) -> None:
    """回测时间仅作为消息正文的前缀，不改变日志头。"""
    _simulation.set(f"{moment:%Y-%m-%d %H:%M}" if moment is not None else "")


def duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分 {seconds % 60} 秒"
    return f"{seconds // 3600} 小时 {seconds % 3600 // 60} 分"
