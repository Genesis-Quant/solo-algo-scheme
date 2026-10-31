"""持有服务端因子分析结果；只在访问、预览或导出时下载。"""

from typing import Any, ClassVar, Generic, Self, TypeVar

import pandas as pd
from query.utils import logger as log

from scheme.data.dolphindb.database.session import download
from scheme.errors import error_text
from scheme.report.base import Report

P = TypeVar("P")


class FactorAnalysisResult(Report, Generic[P]):
    report_kind: ClassVar[str] = "factor"
    data_ref: ClassVar[str] = "factorResult"
    filenames: ClassVar[dict[str, str]] = {
        "execution_statistics": "factor_execution_statistics.parquet",
        "information_coefficient": "factor_information_coefficients.parquet",
        "group_returns": "factor_group_returns.parquet",
        "group_turnover": "factor_group_turnover.parquet",
    }
    # 共同有效样本明细只按需读取：报告不使用它，全市场时却占导出体积的绝大部分。
    tables: ClassVar[tuple[str, ...]] = ("processed_data", *filenames)

    def __init__(self, *, session: Any, parameters: P) -> None:
        self.session = session
        self.parameters = parameters
        self.closed = False

    def read(self, name: str, rows: int | None = None) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(name)
        if self.closed:
            raise RuntimeError("因子分析结果已关闭")
        if rows is not None and (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0):
            raise ValueError("预览行数必须是非负整数")
        reference = f"{self.data_ref}.{name}"
        script = reference if rows is None else f"select top {rows} * from {reference}"
        return download(self.session, script, name)

    @property
    def processed_data(self) -> pd.DataFrame:
        return self.read("processed_data")

    @property
    def execution_statistics(self) -> pd.DataFrame:
        return self.read("execution_statistics")

    @property
    def information_coefficient(self) -> pd.DataFrame:
        return self.read("information_coefficient")

    @property
    def group_returns(self) -> pd.DataFrame:
        return self.read("group_returns")

    @property
    def group_turnover(self) -> pd.DataFrame:
        return self.read("group_turnover")

    def preview(self, name: str, rows: int = 20) -> pd.DataFrame:
        return self.read(name, rows)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.session.close()
        finally:
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("因子分析结果已关闭")
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        try:
            self.close()
        except Exception as error:
            if exception_type is None:
                raise
            log.warning(f"[分析] 异常退出时关闭结果失败：{error_text(error)}")
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(closed={self.closed})"
