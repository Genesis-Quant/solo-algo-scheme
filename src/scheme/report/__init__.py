"""Scheme 内置的前端报告，支持 Notebook 展示及独立 HTML 导出。"""

from .base import Report
from .html import notebook_html, report_html

__all__ = ["Report", "notebook_html", "report_html"]
