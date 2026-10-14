"""ModelAlgo 研究项目基类。"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from scheme.base.internal.algo import Algo
from scheme.base.internal.context import ResearchContext

__all__ = ["ModelAlgo"]

P = TypeVar("P", bound=BaseModel, infer_variance=True)
C = TypeVar("C", bound=ResearchContext[Any], infer_variance=True)


class ModelAlgo(Algo[P, C], Generic[P, C]):
    """在事件回调内更新 signal；其他环节同样可以更新 signal。"""
