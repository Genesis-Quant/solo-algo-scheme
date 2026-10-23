"""结果展示和固定 Parquet 导出；数据来源由具体 Result 管理。"""

import sys
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from runtime.utils import logger as log

from scheme.report.html import report_html


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
        ipython = sys.modules.get("IPython")
        shell = ipython.get_ipython() if ipython is not None else None
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
            log.debug(f"写入 Parquet：{destination}，{len(frame)} 行、{len(frame.columns)} 列")
            frame.to_parquet(temporary, index=False)
            temporary.replace(destination)
            paths.append(destination)
            log.info(
                f"保存报告 {index}/{len(self.filenames)}：{filename}，"
                f"{len(frame)} 行、{len(frame.columns)} 列，{destination.stat().st_size} 字节"
            )
            del frame
        return paths

    def preview(self, name: str, rows: int = 20) -> pd.DataFrame:
        """Jupyter 直接显示返回的 DataFrame。"""
        if name not in self.filenames:
            raise KeyError(name)
        return getattr(self, name).head(rows)
