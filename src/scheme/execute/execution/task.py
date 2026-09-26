from typing import Any

from scheme.execute.research import run as run_research
from scheme.execute.schema import ExecutionTask


def run(data: dict[str, Any], *, input_sha256: str) -> int:
    task = ExecutionTask.model_validate(data)
    return run_research(task, input_sha256=input_sha256)
