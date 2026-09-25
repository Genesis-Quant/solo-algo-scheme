"""研究表单的公共接口。"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

__all__ = ["ReportForm"]


class ReportForm[P: BaseModel](BaseModel, ABC):
    @abstractmethod
    def build(self) -> P:
        """将表单输入转换为当前项目的研究参数。"""
        ...
