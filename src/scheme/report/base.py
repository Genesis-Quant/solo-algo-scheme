"""沿用 Arena 的结果属性和固定 Parquet 文件名；结果不依赖存活会话。"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from runtime.utils import logger as log


@dataclass
class Report:
    filenames: ClassVar[dict[str, str]] = {}
    report_kind: ClassVar[str] = ""

    def show(
        self,
        *,
        height: int | float = 1000,
        theme: Literal["light", "dark"] = "light",
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        """通过 Solo Jupyter 启动桥接提供的 URL iframe 展示交互报告。"""
        message = (
            "report.show() 需要通过 Solo 启动 Jupyter Server，由启动桥接在 IPython shell 上"
            "提供 solo_report_display；独立导出请使用 report.to_html()。"
        )
        try:
            from IPython import get_ipython
        except ImportError as error:
            raise RuntimeError(message) from error

        shell = get_ipython()
        provider = getattr(shell, "solo_report_display", None)
        if not callable(provider):
            raise RuntimeError(message)
        provider(
            self,
            height=height,
            theme=theme,
            annual_trading_days=annual_trading_days,
            risk_free_rate=risk_free_rate,
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
        for index, (name, filename) in enumerate(self.filenames.items(), 1):
            destination = output / filename
            temporary = destination.with_suffix(".parquet.tmp")
            frame = getattr(self, name)
            log.debug("写入 Parquet：{}，{} 行、{} 列", destination, len(frame), len(frame.columns))
            frame.to_parquet(temporary, index=False)
            temporary.replace(destination)
            paths.append(destination)
            log.info("保存报告 {}/{}：{}，{} 行、{} 列，{} 字节",
                     index, len(self.filenames), filename, len(frame), len(frame.columns),
                     destination.stat().st_size)
        return paths

    def preview(self, name: str, rows: int = 20) -> pd.DataFrame:
        """Jupyter 直接显示返回的 DataFrame。"""
        if name not in self.filenames:
            raise KeyError(name)
        return getattr(self, name).head(rows)
