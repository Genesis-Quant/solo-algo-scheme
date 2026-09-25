from typing import Any

from scheme.base import Factor
from scheme.execute.packages import load_component, validate_environment
from scheme.execute.results import check_output, save_run
from scheme.execute.schema import FactorTask


def run(data: dict[str, Any], *, input_sha256: str) -> int:
    task = FactorTask.model_validate(data)
    check_output(task)
    versions = validate_environment(task.environment)
    factor = load_component(task.factor, Factor)
    task.factor.params = task.analysis.factor_params.model_dump(mode="json")
    result = task.analysis.run(factor)
    save_run(task, result, versions, input_sha256=input_sha256)
    return 0
