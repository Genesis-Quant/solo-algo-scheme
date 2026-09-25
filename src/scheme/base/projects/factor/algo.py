from abc import ABC, abstractmethod
from datetime import date
from functools import cached_property

import pandas as pd

from .params import FactorParams

__all__ = ["Factor"]


class Factor[P: FactorParams](ABC):
    """因子计算基类；参数提供研究日期和股票池，不参与 Algo 事件回调。"""

    def __init__(self, params: P) -> None:
        self.params = params

    @cached_property
    def universe(self) -> pd.DataFrame:
        """参数区间的股票池 bool 面板，首次访问时查询。"""
        return self.params.universe.evaluate(self.params.start, self.params.end)

    @abstractmethod
    def compute(self, start: date, end: date) -> pd.DataFrame:
        """返回 [start, end) 的面板；索引为 date、symbol，列为具名因子。

        去极值、标准化、中性化等处理由因子实现包含在返回结果中。
        """
        ...
