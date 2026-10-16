from abc import ABC, abstractmethod
from datetime import date
from random import sample
from typing import Generic, TypeVar

import pandas as pd

from scheme.log import get_logger

from .params import FactorParams

__all__ = ["Factor"]

log = get_logger("scheme.factor")

P = TypeVar("P", bound=FactorParams, infer_variance=True)


def _panel_dates(panel: pd.DataFrame, context: str) -> pd.DatetimeIndex:
    if not isinstance(panel, pd.DataFrame):
        raise ValueError(f"{context}：compute 必须返回 DataFrame")
    if (
        not isinstance(panel.index, pd.MultiIndex)
        or list(panel.index.names) != ["date", "symbol"]
        or not panel.index.is_unique
        or any((codes == -1).any() for codes in panel.index.codes)
    ):
        raise ValueError(f"{context}：因子面板必须使用非空值且唯一的 (date, symbol) 索引")
    if panel.empty:
        raise ValueError(f"{context}：因子面板不能为空或没有因子列")
    if not panel.columns.is_unique:
        raise ValueError(f"{context}：因子列不能重复")
    dates = panel.index.get_level_values("date")
    if not all(isinstance(day, date) for day in dates.unique()):
        raise ValueError(f"{context}：date 索引必须是日期")
    days = pd.DatetimeIndex(dates.unique())
    if days.tz is not None or not days.equals(days.normalize()):
        raise ValueError(f"{context}：date 索引必须是无时区的日历日期")
    return pd.DatetimeIndex(dates)


class Factor(ABC, Generic[P]):
    """因子计算基类；参数只提供股票池及算法设置，不包含日期。

    计算日期由 compute 的闭区间 [start, end] 显式指定，不参与 Algo 事件回调。
    """

    def __init__(self, params: P) -> None:
        self.params = params

    @abstractmethod
    def compute(self, start: date, end: date) -> pd.DataFrame:
        """返回闭区间 [start, end] 的面板；索引为 date、symbol，列为具名因子。

        start == end 为正常单日区间，start > end 非法。
        去极值、标准化、中性化等处理由因子实现包含在返回结果中。
        """
        ...

    def validate(self, start: date, end: date, n_sample: int = 5) -> None:
        """随机抽取结果日期，比较全区间与逐日计算的全部证券、全部列。

        日期采用闭区间 [start, end]，每个样本原样调用 compute(d, d)。
        数值容差固定为 rtol=1e-7、atol=1e-10，非数值和 NULL 位置须一致。
        不一致时抛出 ValueError；通过仅说明抽样一致，并非不存在未来依赖的证明。
        """
        if isinstance(n_sample, bool) or not isinstance(n_sample, int) or n_sample <= 0:
            raise ValueError("n_sample 必须是正整数")
        if start > end:
            raise ValueError("日期区间必须满足 start <= end，采用 [start, end]")

        panel = self.compute(start, end)
        dates = _panel_dates(panel, f"全区间 {start} 至 {end}")
        outside = (dates < pd.Timestamp(start)) | (dates > pd.Timestamp(end))
        if outside.any():
            raise ValueError(
                f"全区间结果日期超出 [{start}, {end}]：{dates[outside].unique().tolist()}"
            )
        days = dates.unique().date.tolist()
        sampled = sample(days, min(n_sample, len(days)))
        log.info("因子抽样校验日期：{}", ", ".join(day.isoformat() for day in sampled))
        # 仅快照抽中的日期，且必须在任何单日 compute 可能改写复用面板之前完成。
        snapshots = [
            (day, panel.loc[dates == pd.Timestamp(day)].copy()) for day in sampled
        ]
        for day, expected in snapshots:
            context = f"样本日期 {day}"
            actual = self.compute(day, day)
            actual_dates = _panel_dates(actual, context)
            if not (actual_dates == pd.Timestamp(day)).all():
                raise ValueError(
                    f"{context}：单日结果含非当天日期 {actual_dates.unique().tolist()}"
                )
            # 日期已完整核对；按证券比较也兼容 date 与午夜 Timestamp 的表示差异。
            expected = expected.droplevel("date")
            actual = actual.droplevel("date")
            for name, left, right in (
                ("索引", expected.index, actual.index),
                ("列", expected.columns, actual.columns),
            ):
                missing = left.difference(right, sort=False)
                extra = right.difference(left, sort=False)
                if len(missing) or len(extra):
                    raise ValueError(
                        f"{context}：{name}不一致，单日缺少 {missing[:5].tolist()}"
                        f"（{len(missing)} 项），额外 {extra[:5].tolist()}（{len(extra)} 项）"
                    )
            try:
                pd.testing.assert_frame_equal(
                    expected.isna(), actual.isna(), check_like=True, check_exact=True,
                )
            except AssertionError as error:
                raise ValueError(f"{context}：NULL 掩码不一致\n{error}") from error
            # NULL 位置已精确比较，逐列只比较非空值，允许 None/NaN/pd.NA 表示不同。
            for column in expected.columns:
                left = expected[column].dropna().infer_objects(copy=False)
                right = actual[column].dropna().infer_objects(copy=False)
                numeric = all(
                    pd.api.types.is_numeric_dtype(values.dtype)
                    and not pd.api.types.is_bool_dtype(values.dtype)
                    for values in (left, right)
                )
                try:
                    pd.testing.assert_series_equal(
                        left, right, check_like=True, check_dtype=False, check_exact=not numeric,
                        check_categorical=False, rtol=1e-7, atol=1e-10,
                    )
                except AssertionError as error:
                    raise ValueError(f"{context}：列 {column!r} 因子值不一致\n{error}") from error
