from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel

from scheme.report.base import Report


@dataclass
class BacktestResult[P: BaseModel](Report):
    report_kind: ClassVar[str] = "backtest"
    trade_details: pd.DataFrame
    daily_positions: pd.DataFrame
    daily_portfolios: pd.DataFrame
    daily_trading_statistics: pd.DataFrame
    allocation_diagnostics: pd.DataFrame
    parameters: P | None = field(default=None, repr=False)

    filenames: ClassVar[dict[str, str]] = {
        "trade_details": "trade_details.parquet",
        "daily_positions": "daily_positions.parquet",
        "daily_portfolios": "daily_portfolios.parquet",
        "daily_trading_statistics": "daily_trading_statistics.parquet",
        "allocation_diagnostics": "allocation_diagnostics.parquet",
    }
