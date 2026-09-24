from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Environment(StrictModel):
    lockfile: Path


class Package(StrictModel):
    package: str
    version: str
    wheel: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    entry: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


class Component(Package):
    params: dict[str, Any] = Field(default_factory=dict)


class Task(StrictModel):
    kind: Literal["factor", "backtest"]
    environment: Environment
    output: Path
