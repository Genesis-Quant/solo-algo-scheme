"""Optimize 项目的研究报告表单接口。"""

from pydantic import BaseModel

from scheme.base.internal.form import ReportForm

__all__ = ["OptimizeReportForm"]


class OptimizeReportForm[P: BaseModel](ReportForm[P]):
    """具体字段与 build() 由该项目后续定义。"""
