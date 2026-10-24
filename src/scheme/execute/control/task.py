from typing import Any

from scheme.execute.checkpoints import Checkpoints
from scheme.execute.research import run as run_research
from scheme.execute.schema import ControlTask


def run(data: dict[str, Any], *, checkpoints: Checkpoints | None = None) -> int:
    task = ControlTask.model_validate(data)
    return run_research(task, checkpoints=checkpoints)
