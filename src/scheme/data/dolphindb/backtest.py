"""保留 Runtime 回测 API，仅将已知回测模块内联到当前会话。"""

import re
from collections.abc import Mapping
from functools import wraps
from importlib.resources import files
from inspect import signature
from typing import Any

from runtime.apps.backtest import api as _runtime

_BACKTEST_SCRIPT = files(__package__).joinpath("arena_backtest.dos")
_SIGNATURE = signature(_runtime.run_backtest)
_FUNCTIONS = frozenset({
    "getParams", "getParam", "getIndustry", "getTradeDates", "getHistoryData",
    "getLastData", "getMinuteHistory", "order_target", "order_target_value",
    "build_backtest_message", "run_backtest", "run_snapshot_backtest",
    "standardize_return_summary", "backtest_order_quote", "create_backtest_engine",
    "read_stock_snapshots", "build_minute_bars", "build_stock_snapshot_message",
    "snapshot_callback",
})
# 仅区分不可改写的文本与标识符，不解析或重新生成 DOS。
_TOKENS = re.compile(
    r"(?P<comment>//[^\r\n]*|/\*[\s\S]*?(?:\*/|\Z))"
    r'''|(?P<literal>"(?:\\[\s\S]|[^"\\])*(?:"|\Z)|'(?:\\[\s\S]|[^'\\])*(?:'|\Z)|`[\w:]*)'''
    r"|(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)"
    r"|(?P<space>\s+)|(?P<other>[\s\S])"
)


def _code_tokens(script: str) -> list[re.Match[str]]:
    return [
        token for token in _TOKENS.finditer(script)
        if token.lastgroup not in {"comment", "space"}
    ]


def _rewrite_script(script: str) -> str:
    tokens = _code_tokens(script)
    for index, token in enumerate(tokens):
        if token.lastgroup == "name" and token.group() == "use":
            module = tokens[index + 1].group() if index + 1 < len(tokens) else ""
            if module != "query" or tokens[index + 1].lastgroup != "name":
                raise ValueError(f"Scheme 回测仅允许 use query，不支持 use {module}")

    def replace(token: re.Match[str]) -> str:
        value = token.group()
        if token.lastgroup == "name" and value.startswith("backtest::"):
            name = value.removeprefix("backtest::")
            if name not in _FUNCTIONS:
                raise ValueError(f"Scheme 未内联回测函数：{value}")
            return name
        return value

    return _TOKENS.sub(replace, script)


class _InlineBacktestSession:
    """仅拦截 Runtime 独立 bootstrap；结果对象继续持有该代理。"""

    def __init__(self, session: Any) -> None:
        self.raw_session = session
        self._loaded = False

    def run(self, script: str, *args: Any, **kwargs: Any) -> Any:
        bootstrap = script.strip().removesuffix(";").rstrip()
        if bootstrap == "use factor":
            return None
        if bootstrap == "use backtest":
            if not self._loaded:
                result = self.raw_session.run(
                    _BACKTEST_SCRIPT.read_text(encoding="utf-8"), *args, **kwargs,
                )
                self._loaded = True
                return result
            return None
        return self.raw_session.run(_rewrite_script(script), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_session, name)


@wraps(_runtime.run_backtest)
def backtest(*args: Any, **kwargs: Any) -> Any:
    bound = _SIGNATURE.bind(*args, **kwargs)
    # 在 Runtime 拼接函数源码前校验，不能把用户的 use factor 当作 bootstrap。
    callbacks = bound.arguments["callbacks"]
    scripts = list(callbacks.values()) if isinstance(callbacks, Mapping) else []
    scripts.append(bound.arguments.get("utils", ""))
    for script in scripts:
        if isinstance(script, str):
            _rewrite_script(script)

    session = bound.arguments.get("session")
    owns_session = session is None
    if owns_session:
        session = _runtime.create_session(max_time=_runtime.BACKTEST_SESSION_MAX_TIME)
    proxy = session if isinstance(session, _InlineBacktestSession) else _InlineBacktestSession(session)
    bound.arguments["session"] = proxy
    try:
        return _runtime.run_backtest(*bound.args, **bound.kwargs)
    except Exception:
        # Runtime 收到显式 session 后不再负责其异常生命周期。
        if owns_session:
            session.close()
        raise


__all__ = ["backtest"]
