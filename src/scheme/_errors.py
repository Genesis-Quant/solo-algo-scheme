"""异常文字脱敏，不改变日志实例、格式或级别。"""

import os
import re
from traceback import format_exception


def error_text(error: BaseException, *secrets: str, traceback: bool = False) -> str:
    text = ("".join(format_exception(error)) if traceback
            else f"{type(error).__name__}: {error}")
    values = {*secrets, *(value for key, value in os.environ.items()
                         if any(part in key.upper() for part in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")))}
    for value in sorted(filter(None, values), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return re.sub(r"(https?://)[^/\s@]+@", r"\1[REDACTED]@", text).rstrip()
