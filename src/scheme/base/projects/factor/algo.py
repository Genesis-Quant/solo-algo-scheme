from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from functools import wraps
from inspect import isabstract
from random import sample
from typing import ClassVar, Generic, TypeVar

import pandas as pd
from backtest import DosVar
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.config import DolphinSettings

from .params import FactorParams
from .preprocess import preprocess

__all__ = ["Factor"]

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


def _validate_result(
    panel: pd.DataFrame, columns: Sequence[str], context: str,
) -> pd.DataFrame:
    """严格核对全部输出列，并将 time/code 平表转为既有面板。"""
    if not isinstance(panel, pd.DataFrame):
        raise ValueError(f"{context}：compute 必须返回 DataFrame")
    if not panel.columns.is_unique:
        raise ValueError(f"{context}：因子列不能重复")
    if panel.empty:
        raise ValueError(f"{context}：因子面板不能为空或没有因子列")
    raw = not isinstance(panel.index, pd.MultiIndex) and {"time", "code"} <= set(panel.columns)
    expected = ["time", "code", *columns] if raw else list(columns)
    missing = [column for column in expected if column not in panel.columns]
    extra = [column for column in panel.columns if column not in expected]
    if missing or extra:
        raise ValueError(f"{context}：因子列不一致，缺少 {missing}，额外 {extra}")
    if raw:
        panel = panel.set_index(["time", "code"]).rename_axis(["date", "symbol"])
    _panel_dates(panel, context)
    return panel


class Factor(ABC, Generic[P]):
    """因子计算基类；参数只提供股票池及算法设置，不包含日期。

    计算日期由 compute 的闭区间 [start, end] 显式指定，不参与 Algo 事件回调。
    具体类必须显式声明 columns，compute 不得返回未声明的因子或暴露列。
    """

    columns: ClassVar[list[str] | tuple[str, ...]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not isabstract(cls) or "columns" in cls.__dict__:
            columns = getattr(cls, "columns", None)
            if not isinstance(columns, (list, tuple)) or not columns or any(
                not isinstance(column, str) or not column.strip() or "\0" in column
                for column in columns
            ):
                raise TypeError(f"{cls.__name__}.columns 必须显式声明为非空有效字符串 list/tuple")
            if len(set(columns)) != len(columns):
                raise ValueError(f"{cls.__name__}.columns 因子列不能重复")
            if set(columns) & {
                "time", "code", "date", "symbol", "stock_pool_member", "circ_mv", "industry_l0",
            }:
                raise ValueError(f"{cls.__name__}.columns 因子列不能与键、暴露或分析保留列重名")
        compute = cls.__dict__.get("compute")
        if (
            compute is not None
            and not getattr(compute, "__isabstractmethod__", False)
            and getattr(compute, "_scheme_checked_compute", None) is not compute
        ):
            @wraps(compute)
            def checked_compute(self, *args, **kwargs):
                name = f"{type(self).__module__}.{type(self).__qualname__}"
                start = kwargs.get("start", args[0] if args else None)
                end = kwargs.get("end", args[1] if len(args) > 1 else None)
                log.info(
                    "因子计算开始：{}，区间 {} 至 {}，因子列 {}",
                    name, start if isinstance(start, date) else "未知",
                    end if isinstance(end, date) else "未知", self.columns,
                )
                try:
                    panel = compute(self, *args, **kwargs)
                    panel = _validate_result(panel, self.columns, f"{name}.compute")
                    log.opt(lazy=True).info(
                        "因子计算完成：{}，{} 行，{} 个日期，{} 只证券；"
                        "各因子 NULL 数 {}",
                        lambda: name, lambda: len(panel),
                        lambda: panel.index.get_level_values("date").nunique(),
                        lambda: panel.index.get_level_values("symbol").nunique(),
                        lambda: panel.isna().sum().to_dict(),
                    )
                    return panel
                except Exception as error:
                    log.error(
                        "因子计算失败：{}，范围 {} 至 {}；{}",
                        name, start, end, error_text(error),
                    )
                    raise

            checked_compute._scheme_checked_compute = checked_compute
            cls.compute = checked_compute

    def __init__(self, params: P) -> None:
        self.params = params

    @abstractmethod
    def compute(self, start: date, end: date) -> pd.DataFrame:
        """返回闭区间 [start, end] 的 columns 因子，可用 (date, symbol) 面板或 time/code 平表。

        start == end 为正常单日区间，start > end 非法。
        去极值、标准化、中性化等处理由因子实现包含在返回结果中。
        """
        ...

    def preprocess(
        self,
        data: pd.DataFrame | DosVar,
        *,
        session=None,
        market_value_column: str = "circ_mv",
        industry_column: str = "industry_l0",
        settings: DolphinSettings | None = None,
    ) -> pd.DataFrame:
        """在 compute 返回前调用 DOS 去极值、标准化和市值/动态行业中性化。

        接受 time/code 普通列或 (date, symbol) 面板；DosVar 必须显式传 session。
        因子列来自 self.columns；暴露按输入日期和证券独立查询。无效或退化记录
        保留为 NULL，不生成分组；返回唯一、排序的 (date, symbol) 数值因子面板。
        """
        return preprocess(
            data, session=session, columns=self.columns,
            market_value_column=market_value_column, industry_column=industry_column,
            settings=settings,
        )

    def validate(self, start: date, end: date, n_sample: int = 5) -> None:
        """随机抽取结果日期，比较全区间与逐日计算的全部证券、全部列。

        日期采用闭区间 [start, end]，每个样本原样调用 compute(d, d)。
        数值容差固定为 rtol=1e-7、atol=1e-10，非数值和 NULL 位置须一致。
        不一致时抛出 ValueError；通过仅说明抽样一致，并非不存在未来依赖的证明。
        """
        factor_name = f"{type(self).__module__}.{type(self).__qualname__}"
        log.info(
            "因子抽样校验开始：{}，区间 {} 至 {}，因子列 {}",
            factor_name, start if isinstance(start, date) else "未知",
            end if isinstance(end, date) else "未知", self.columns,
        )
        sample_index, sample_total, sample_day = 0, 0, None
        try:
            if isinstance(n_sample, bool) or not isinstance(n_sample, int) or n_sample <= 0:
                raise ValueError("n_sample 必须是正整数")
            if start > end:
                raise ValueError("日期区间必须满足 start <= end，采用 [start, end]")

            context = f"全区间 {start} 至 {end}"
            try:
                panel = self.compute(start, end)
            except ValueError as error:
                raise ValueError(f"{context}：{error}") from error
            dates = _panel_dates(panel, context)
            outside = (dates < pd.Timestamp(start)) | (dates > pd.Timestamp(end))
            if outside.any():
                raise ValueError(
                    f"全区间结果日期超出 [{start}, {end}]：{dates[outside].unique().tolist()}"
                )
            days = dates.unique().date.tolist()
            log.opt(lazy=True).info(
                "因子抽样校验全区间结果：{}，{} 行、{} 列、{} 个日期、{} 只证券",
                lambda: factor_name, lambda: len(panel), lambda: len(panel.columns),
                lambda: len(days), lambda: panel.index.get_level_values("symbol").nunique(),
            )
            sampled = sample(days, min(n_sample, len(days)))
            sample_total = len(sampled)
            log.info("因子抽样校验日期：{}", ", ".join(day.isoformat() for day in sampled))
            # 仅快照抽中的日期，且必须在任何单日 compute 可能改写复用面板之前完成。
            snapshots = [
                (day, panel.loc[dates == pd.Timestamp(day)].copy()) for day in sampled
            ]
            for sample_index, (day, expected) in enumerate(snapshots, 1):
                sample_day = day
                context = f"样本日期 {day}"
                log.debug(
                    "因子抽样校验样本 {}/{} 开始：{}，日期 {}",
                    sample_index, sample_total, factor_name, day,
                )
                try:
                    actual = self.compute(day, day)
                except ValueError as error:
                    raise ValueError(f"{context}：{error}") from error
                log.debug(
                    "因子抽样校验样本 {}/{} 计算完成：{}，日期 {}",
                    sample_index, sample_total, factor_name, day,
                )
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
                log.info(
                    "因子抽样校验样本 {}/{} 比较通过：{}，日期 {}",
                    sample_index, sample_total, factor_name, day,
                )
            log.info(
                "因子抽样校验全部通过：{}，{} 个样本",
                factor_name, sample_total,
            )
        except Exception as error:
            log.error(
                "因子校验失败：{}，样本 {}/{}，日期={}；{}",
                factor_name, sample_index, sample_total, sample_day, error_text(error),
            )
            raise
