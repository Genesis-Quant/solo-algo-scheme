from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scheme.base import BacktestParameters, FactorAnalysisParams
from scheme.execute.strategy.assembly import AlgoComponents


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
    kind: Literal["factor", "model", "optimize", "control", "execution", "strategy"]
    environment: Environment
    output: Path


class FactorTask(Task):
    kind: Literal["factor"] = "factor"
    factor: Component
    analysis: FactorAnalysisParams


class AlgoTask(Task):
    algos: AlgoComponents[Package]
    context: str | None = Field(default=None, pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    backtest: BacktestParameters

    @model_validator(mode="after")
    def require_research_component(self) -> Self:
        if self.kind in {"model", "optimize", "control", "execution"} and getattr(self.algos, self.kind) is None:
            raise ValueError(f"{self.kind} 研究任务必须提供当前环节的 Algo")
        return self


class ModelTask(AlgoTask):
    kind: Literal["model"] = "model"


class OptimizeTask(AlgoTask):
    kind: Literal["optimize"] = "optimize"


class ControlTask(AlgoTask):
    kind: Literal["control"] = "control"


class ExecutionTask(AlgoTask):
    kind: Literal["execution"] = "execution"


class StrategyTask(AlgoTask):
    kind: Literal["strategy"] = "strategy"
