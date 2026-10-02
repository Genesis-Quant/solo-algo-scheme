from typing import Any

from scheme.base import Factor
from scheme.execute.factor.api import analyze_factors
from scheme.execute.packages import load_component, validate_environment
from scheme.execute.results import check_output, save_run
from scheme.execute.schema import FactorTask
from scheme.execute.strategy.assembly import parameter_type, project_parameters


def run(data: dict[str, Any], *, input_sha256: str) -> int:
    task = FactorTask.model_validate(data)
    check_output(task)
    versions = validate_environment(task.environment)
    factor = load_component(task.factor, Factor)
    shared = {name: getattr(task.analysis, name) for name in ("start", "end", "universe")}
    params = project_parameters(parameter_type(factor), task.factor.params, overrides=shared)
    task.factor.params = params.model_dump(mode="json")
    result = analyze_factors(factor(params), task.analysis)
    save_run(task, result, versions, input_sha256=input_sha256)
    return 0
