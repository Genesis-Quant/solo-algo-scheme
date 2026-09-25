from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from scheme.report.base import Report

if TYPE_CHECKING:
    from scheme.base import FactorAnalysisParams


@dataclass
class FactorAnalysisResult(Report):
    processed_data: pd.DataFrame
    execution_statistics: pd.DataFrame
    information_coefficient: pd.DataFrame
    group_returns: pd.DataFrame
    group_turnover: pd.DataFrame
    parameters: "FactorAnalysisParams | None" = field(default=None, repr=False)

    report_kind: ClassVar[str] = "factor"

    filenames: ClassVar[dict[str, str]] = {
        "processed_data": "factor_processed.parquet",
        "execution_statistics": "factor_execution_statistics.parquet",
        "information_coefficient": "factor_information_coefficients.parquet",
        "group_returns": "factor_group_returns.parquet",
        "group_turnover": "factor_group_turnover.parquet",
    }
