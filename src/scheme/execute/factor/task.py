from typing import Any

from runtime.utils import logger as log

from scheme.base import Factor
from scheme.execute.factor.api import analyze_factors
from scheme.execute.packages import installed_versions, load_component
from scheme.execute.results import check_output, save_run
from scheme.execute.schema import FactorTask
from scheme.execute.strategy.assembly import parameter_type, project_parameters


def run(data: dict[str, Any]) -> int:
    task = FactorTask.model_validate(data)
    check_output(task)
    versions = installed_versions()
    factor = load_component(task.factor, Factor)
    log.info("加载因子：{} {}（{}）", task.factor.package, task.factor.version, task.factor.entry)
    shared = {"universe": task.analysis.universe}
    params = project_parameters(parameter_type(factor), task.factor.params, overrides=shared)
    task.factor.params = params.model_dump(mode="json")
    result = analyze_factors(factor(params), task.analysis)
    save_run(task, result, versions)
    return 0
