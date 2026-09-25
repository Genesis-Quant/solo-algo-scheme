"""ModelAlgo 研究项目基类。"""

from typing import Any

from pydantic import BaseModel

from scheme.base.internal.algo import Algo
from scheme.base.internal.context import ResearchContext

__all__ = ["ModelAlgo"]


class ModelAlgo[P: BaseModel, C: ResearchContext[Any]](Algo[P, C]):
    """在事件回调内更新 signal；其他环节同样可以更新 signal。"""
