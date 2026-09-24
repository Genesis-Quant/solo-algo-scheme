from typing import Any

from scheme import Factor
from scheme.apps.factor import analyze_factors
from scheme.apps.factor.schema import FactorTask
from scheme.utils.manage import check_output, save_run
from scheme.utils.packages import instantiate, validate_environment


def run(data: dict[str, Any], *, input_sha256: str) -> int:
    task = FactorTask.model_validate(data)
    check_output(task)
    versions = validate_environment(task.environment)
    task.factor.params.update(
        task.analysis.model_dump(mode="json", include={"start", "end", "universe"})
    )
    factor = instantiate(task.factor, Factor)
    task.factor.params = factor.params.model_dump(mode="json")
    result = analyze_factors(factor, task.analysis)
    save_run(task, result, versions, input_sha256=input_sha256)
    return 0
