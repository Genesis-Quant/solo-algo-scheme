"""通用工具；数据、算法定义、执行及报告分别由对应子包导出。"""

import os
import sys

if "loguru" not in sys.modules and "LOGURU_AUTOINIT" not in os.environ:
    os.environ["LOGURU_AUTOINIT"] = "False"
    from loguru import logger

    del os.environ["LOGURU_AUTOINIT"]
    logger.add(sys.stdout, diagnose=False)
else:
    from loguru import logger

__all__ = ["logger"]
