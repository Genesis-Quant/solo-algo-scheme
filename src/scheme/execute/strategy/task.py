from typing import Any

from scheme.execute.research import run as run_research
from scheme.execute.schema import StrategyTask


def run(data: dict[str, Any]) -> int:
    task = StrategyTask.model_validate(data)
    return run_research(task)
