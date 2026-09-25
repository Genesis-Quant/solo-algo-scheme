"""沿用 Arena 的结果属性和固定 Parquet 文件名；结果不依赖存活会话。"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd


@dataclass
class Report:
    filenames: ClassVar[dict[str, str]] = {}
    report_kind: ClassVar[str] = ""

    def show(
        self,
        *,
        height: int = 1000,
        theme: Literal["light", "dark"] = "light",
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        """在 Jupyter 展示与前端相同的完整交互报告，不需要前端服务。"""
        from IPython.display import display

        from scheme.report import notebook_html

        display(
            {
                "text/html": notebook_html(
                    self,
                    height=height,
                    theme=theme,
                    annual_trading_days=annual_trading_days,
                    risk_free_rate=risk_free_rate,
                ),
                "text/plain": "Scheme 交互研究报告",
            },
            raw=True,
        )

    def to_html(
        self,
        *,
        theme: Literal["light", "dark"] = "light",
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.0,
    ) -> str:
        """返回内含页面、图表引擎和数据的独立 HTML，可保存后离线打开。"""
        from scheme.report import report_html

        return report_html(
            self,
            theme=theme,
            annual_trading_days=annual_trading_days,
            risk_free_rate=risk_free_rate,
        )

    def save(self, output: Path) -> list[Path]:
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, filename in self.filenames.items():
            destination = output / filename
            temporary = destination.with_suffix(".parquet.tmp")
            getattr(self, name).to_parquet(temporary, index=False)
            temporary.replace(destination)
            paths.append(destination)
        return paths

    def preview(self, name: str, rows: int = 20) -> pd.DataFrame:
        """Jupyter 直接显示返回的 DataFrame。"""
        if name not in self.filenames:
            raise KeyError(name)
        if rows < 0:
            raise ValueError("rows 不能为负数")
        return getattr(self, name).head(rows)
