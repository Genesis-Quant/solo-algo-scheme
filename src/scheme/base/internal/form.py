"""UI 表单只负责把简化输入转换成具体分析参数。"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

__all__ = ["ReportForm"]


class ReportForm[P: BaseModel](BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @abstractmethod
    def build(self) -> P:
        """构造并校验分析参数；不执行研究。"""
        ...
