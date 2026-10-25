"""单次回测的原始收盘历史；只允许读取模拟时钟已经公开的数据。"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scheme.data.dolphindb.symbols import source_symbol

if TYPE_CHECKING:
    from runtime.apps.query import FactorQuery

    from scheme.config import DolphinSettings

__all__ = ["History", "current_history", "history_scope"]

_FIELDS = ("close", "adj_factor")
_HISTORY: ContextVar["History | None"] = ContextVar("backtest_history", default=None)


def current_history() -> "History | None":
    return _HISTORY.get()


@contextmanager
def history_scope(history: "History | None") -> Iterator[None]:
    token = _HISTORY.set(history)
    try:
        yield
    finally:
        _HISTORY.reset(token)


def _date(value: Any, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{name} 不能为空")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{name} 不能是 NaT")
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _codes(codes: Sequence[str]) -> tuple[str, ...]:
    if codes is None or isinstance(codes, str):
        raise TypeError("codes 必须是显式证券列表")
    if any(not isinstance(code, str) or not code.strip() for code in codes):
        raise ValueError("codes 必须包含非空证券代码")
    return tuple(dict.fromkeys(source_symbol(code.strip()) for code in codes))


class History:
    def __init__(
        self,
        session: Any,
        settings: "DolphinSettings",
        start: Any,
        end: Any,
        codes: Sequence[str],
        lookback_days: int = 180,
        *,
        clock: Callable[[], Any],
    ) -> None:
        self.session = session
        self.settings = settings
        self.start = _date(start, "start")
        self.end = _date(end, "end")
        if self.start >= self.end:
            raise ValueError("start 必须早于 end（不含）")
        if lookback_days < 0:
            raise ValueError("lookback_days 不能小于 0")
        self.lookback_days = max(60, lookback_days)
        # 缓存仅属于本次回测，且有界；旧 query 的 get→query 不再重复规范化代码。
        self._cached_codes = lru_cache(maxsize=32)(_codes)
        self.codes = self._normalize_codes(codes)
        self.clock = clock
        self._start = self.start - timedelta(days=self.lookback_days)
        self._code_set = frozenset(self.codes)
        self._data: pd.DataFrame | None = None
        self._times = pd.DatetimeIndex([])
        self._time_positions = np.array([], dtype=np.intp)
        self._row_codes = np.array([], dtype=np.intp)
        self._code_slots: dict[str, int] = {}
        self.calendar = pd.DatetimeIndex([])

    def _normalize_codes(self, codes: Sequence[str]) -> tuple[str, ...]:
        if codes is None or isinstance(codes, str):
            raise TypeError("codes 必须是显式证券列表")
        return self._cached_codes(tuple(codes))

    def prepare(self) -> None:
        """一次性读取 [start-lookback, end)，不以尚未推进的时钟判断预加载。"""
        if self._data is not None:
            return
        if not self.codes:
            self._data = pd.DataFrame(
                {
                    "time": pd.Series(dtype="datetime64[ns]"),
                    "code": pd.Series(dtype=str),
                    **{field: pd.Series(dtype=float) for field in _FIELDS},
                }
            )
            return
        # 查询模块也会使用 current_history；初始化时必须绕过该兼容入口。
        from scheme.data import query

        with (
            history_scope(None),
            query(
                {
                    "start_date": self._start.date().isoformat(),
                    "end_date": (self.end - timedelta(days=1)).date().isoformat(),
                    "codes": list(self.codes),
                    "factors": list(_FIELDS),
                },
                session=self.session,
                settings=self.settings,
            ) as result,
        ):
            data = result.data.copy(deep=True)
            calendar = pd.DatetimeIndex(result.calendar).copy()
        self.calendar = calendar[(calendar >= self._start) & (calendar < self.end)]
        data["time"] = pd.to_datetime(data.time)
        data["code"] = data.code.map(source_symbol)
        self._data = (
            data.loc[
                data.time.ge(self._start) & data.time.lt(self.end) & data.code.isin(self.codes),
                ["time", "code", *_FIELDS],
            ]
            .sort_values(["code", "time"])
            .reset_index(drop=True)
        )
        # 原始行按最终输出顺序存储，时间索引仅保存行位置，窗口查询不再排序 DataFrame。
        self._row_codes, codes = pd.factorize(self._data.code, sort=False)
        self._code_slots = {code: index for index, code in enumerate(codes)}
        self._time_positions = np.argsort(self._data.time.to_numpy(), kind="stable")
        self._times = pd.DatetimeIndex(self._data.time.iloc[self._time_positions])

    def _now(self) -> pd.Timestamp | None:
        value = self.clock()
        if value is None or pd.isna(value):
            return None
        now = pd.Timestamp(value)
        return now.tz_localize(None) if now.tzinfo is not None else now

    @staticmethod
    def _check_visible(end: pd.Timestamp, now: pd.Timestamp, include_today: bool) -> None:
        today = now.normalize()
        if end > today:
            raise ValueError("历史查询不能读取模拟时点之后的数据")
        if end == today and (not include_today or now.hour < 15):
            raise ValueError("当日收盘数据只允许在 15:00 后显式 include_today=True 读取")

    def query(
        self,
        start_date: Any,
        end_date: Any,
        codes: Sequence[str],
        fields: Sequence[str] = _FIELDS,
        include_today: bool = False,
    ) -> pd.DataFrame:
        """按闭区间自然日返回独立副本；不扩窗、不填充、不计算跨窗收益。"""
        start = _date(start_date, "start_date")
        end = _date(end_date, "end_date")
        if start > end:
            raise ValueError("start_date 不能晚于 end_date")
        now = self._now()
        if now is None:
            raise ValueError("模拟时钟为 NaT，不能读取历史")
        self._check_visible(end, now, include_today)
        if isinstance(fields, str) or not set(fields).issubset(_FIELDS):
            raise ValueError("历史仅预加载 close 和 adj_factor")
        fields = tuple(dict.fromkeys(fields))
        selected_codes = self._normalize_codes(codes)
        if self._data is None:
            raise RuntimeError("历史尚未 prepare")
        if start < self._start or end >= self.end:
            raise ValueError("历史查询超出预加载日期范围")
        if not set(selected_codes).issubset(self._code_set):
            raise ValueError("历史查询包含未预加载的证券")
        left = self._times.searchsorted(start, side="left")
        right = self._times.searchsorted(end + timedelta(days=1), side="left")
        positions = self._time_positions[left:right]
        if len(selected_codes) != len(self._code_set):
            chosen = np.zeros(len(self._code_slots), dtype=bool)
            slots = [self._code_slots[code] for code in selected_codes if code in self._code_slots]
            chosen[slots] = True
            positions = positions[chosen[self._row_codes[positions]]]
        # 按原始行位置排列即为 code/time；只排序整数，不改动价格或缺值。
        window = self._data.iloc[np.sort(positions)]
        if fields != _FIELDS:
            window = window.loc[:, ["time", "code", *fields]]
        return window.reset_index(drop=True).copy(deep=True)

    def get(self, request: "FactorQuery") -> pd.DataFrame | None:
        """仅加速无 DSL 的原始价格查询；None 交还正常 QueryResult 路径。"""
        now = self._now()
        if now is None:
            return None
        start = _date(request.start_date, "start_date")
        end = _date(request.end_date, "end_date")
        # 未来请求不能因缓存 miss 回落到数据库读取；DSL 依赖的当日收盘同样受限。
        if end > now.normalize():
            raise ValueError("历史查询不能读取模拟时点之后的数据")
        if end == now.normalize() and set(request.source_factors()).intersection(_FIELDS):
            self._check_visible(end, now, include_today=True)
        if (
            request.derivatives
            or request.filters
            or request.lookback != timedelta(0)
            or not request.codes
            or not request.factors
            or not set(request.factors).issubset(_FIELDS)
        ):
            return None
        codes = self._normalize_codes(request.codes)
        if (
            self._data is None
            or start < self._start
            or end >= self.end
            or not set(codes).issubset(self._code_set)
        ):
            return None
        return self.query(start, end, codes, request.factors, include_today=True)
