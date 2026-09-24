"""沿用 Arena 的结果属性和固定 Parquet 文件名；结果不依赖存活会话。"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd


@dataclass
class Report:
    filenames: ClassVar[dict[str, str]] = {}

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
