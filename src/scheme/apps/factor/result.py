from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from scheme.utils.result import Report


@dataclass
class FactorAnalysisResult(Report):
    processed_data: pd.DataFrame
    execution_statistics: pd.DataFrame
    information_coefficient: pd.DataFrame
    group_returns: pd.DataFrame
    group_turnover: pd.DataFrame

    filenames: ClassVar[dict[str, str]] = {
        "processed_data": "factor_processed.parquet",
        "execution_statistics": "factor_execution_statistics.parquet",
        "information_coefficient": "factor_information_coefficients.parquet",
        "group_returns": "factor_group_returns.parquet",
        "group_turnover": "factor_group_turnover.parquet",
    }
