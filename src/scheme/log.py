"""研究进程的统一日志：Scheme、backtest 库、研究项目与 Arena 输出同一格式。

默认 INFO 只显示研究进度、结果和异常；Arena 每次查询的会话与建表步骤归入 DEBUG。
环境变量 SCHEME_LOG_LEVEL 或 configure(level=...) 可切换为 DEBUG、WARNING 等级别。
研究项目可直接使用标准库 logging.getLogger(__name__)，输出会接入这里的格式与级别；
其他第三方库的标准库日志只显示 WARNING 及以上。
"""

import logging
import os
import re
import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TextIO

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger, Record

__all__ = ["configure", "get_logger"]

LEVEL_VARIABLE = "SCHEME_LOG_LEVEL"
FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {extra[component]} | {extra[simulation]}{message}"
)
# 回测按批次反复查询，Arena 的逐步会话日志会淹没研究进度；警告和错误仍正常输出。
_DETAIL_SOURCES = ("runtime.apps.query", "runtime.database.session")
# 标准库日志按所选级别输出的来源：研究链路、Notebook 和研究项目包（含模板包名）。
_RESEARCH_SOURCES = re.compile(
    r"(?:backtest|scheme|__main__"
    r"|(?:factor|model|optimize|control|execution|strategy)(?:_[0-9a-f]{4})?)(?:\.|$)"
)
_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}
_simulation: ContextVar[str] = ContextVar("scheme_log_simulation", default="")
_current: tuple[str, TextIO] | None = None


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
        logger.bind(component=record.name).opt(exception=record.exc_info).log(level, message)


def _component(name: str | None) -> str:
    parts = (name or "-").split(".")
    if parts[0] == "runtime" and len(parts) > 1:
        return "arena." + (parts[2] if parts[1] == "apps" and len(parts) > 2 else parts[1])
    return name or "-"


def _patch(record: "Record") -> None:
    extra = record["extra"]
    extra.setdefault("component", _component(record["name"]))
    moment = _simulation.get()
    extra["simulation"] = f"{moment} | " if moment else ""


def _accept(threshold: int):
    debug = logger.level("DEBUG").no
    warning = logger.level("WARNING").no

    def accept(record: "Record") -> bool:
        if record["level"].no < warning and (record["name"] or "").startswith(_DETAIL_SOURCES):
            return threshold <= debug
        return True

    return accept


def configure(level: str | None = None, *, stream: TextIO | None = None) -> None:
    """设置当前进程的日志级别和输出流；相同设置重复调用不会重建处理器。

    level 省略时读取 SCHEME_LOG_LEVEL，默认 INFO。显式传入的无效级别直接报错；
    环境变量无效时不阻断导入，回退为 INFO 并输出一条警告。
    """
    # Arena 首次导入日志模块时会重置 loguru，必须在接管输出前完成。
    import runtime.utils.logging  # noqa: F401

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
    # stdout 留给 CLI 的 JSON 等程序输出；Notebook 内核中写 stdout，避免被显示为错误流。
    in_kernel = type(sys.stdout).__module__.startswith("ipykernel")
    target = stream or (sys.stdout if in_kernel else sys.stderr)
    global _current
    if _current != (name, target):
        logger.remove()
        logger.configure(patcher=_patch)
        logger.add(
            target, level=threshold, format=FORMAT, filter=_accept(threshold),
            colorize=None, backtrace=False, diagnose=False,
        )
        standard = logging.getLevelName(name)
        logging.basicConfig(
            handlers=[_Intercept()], level=standard if isinstance(standard, int) else threshold,
            force=True,
        )
        _current = (name, target)
    if invalid:
        get_logger("scheme").warning("{} 无效：{}，已按 INFO 输出日志", LEVEL_VARIABLE, invalid)


def get_logger(component: str) -> "Logger":
    """返回带组件名的 loguru logger，供 Scheme 内部模块使用。"""
    return logger.bind(component=component)


def mark_simulation(moment: Any | None) -> None:
    """回测推进时记录模拟时间；None 表示离开回测。"""
    _simulation.set(f"{moment:%Y-%m-%d %H:%M}" if moment is not None else "")


def duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分 {seconds % 60} 秒"
    return f"{seconds // 3600} 小时 {seconds % 3600 // 60} 分"
