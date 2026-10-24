from typing import Any

from runtime.utils import logger as log

from scheme.base import Factor
from scheme.execute.checkpoints import Checkpoints
from scheme.execute.factor.api import analyze_factors
from scheme.execute.factor.result import FactorAnalysisResult
from scheme.execute.packages import installed_versions, load_component
from scheme.execute.results import check_output, resume_run, save_run
from scheme.execute.schema import FactorTask
from scheme.execute.strategy.assembly import parameter_type, project_parameters


def run(data: dict[str, Any], *, checkpoints: Checkpoints | None = None) -> int:
    task = FactorTask.model_validate(data)
    check_output(task)
    versions = installed_versions()
    factor = load_component(task.factor, Factor)
    log.info(f"加载因子：{task.factor.package} {task.factor.version}（{task.factor.entry}）")
    shared = {"universe": task.analysis.universe, "groups": task.analysis.groups}
    params = project_parameters(parameter_type(factor), task.factor.params, overrides=shared)
    task.factor.params = params.model_dump(mode="json")
    if resume_run(task, FactorAnalysisResult, versions, checkpoints):
        return 0
    with analyze_factors(factor(params), task.analysis, checkpoints=checkpoints) as result:
        save_run(task, result, versions, checkpoints)
    return 0
