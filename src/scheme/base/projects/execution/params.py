"""Execution 项目的算法参数基类。"""

from pydantic import BaseModel, ConfigDict

__all__ = ["ExecutionParams"]


class ExecutionParams(BaseModel):
    """从统一策略参数中读取当前算法声明的字段。"""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
