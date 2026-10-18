"""会话内 DOS 函数定义，不加载或部署数据库模块。"""

from importlib.resources import files
from typing import Literal


def factor_script(kind: Literal["analysis", "preprocess"]) -> str:
    if kind not in {"analysis", "preprocess"}:
        raise ValueError(f"未知因子脚本：{kind}")
    root = files(__package__)
    return "\n".join(
        root.joinpath(name).read_text(encoding="utf-8")
        for name in ("factor_helpers.dos", f"factor_{kind}.dos")
    )
