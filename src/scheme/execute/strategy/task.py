from importlib.metadata import distribution
from typing import Any

from runtime.utils import logger as log

from scheme.base import Algo, ResearchContext
from scheme.execute.checkpoints import Checkpoints
from scheme.execute.packages import installed_versions, load_component, load_entry
from scheme.execute.results import check_output, resume_run, save_run
from scheme.execute.schema import AlgoTask

from .api import run_backtest
from .assembly import Strategy, context_type
from .result import BacktestResult


def run(data: dict[str, Any], *, checkpoints: Checkpoints | None = None) -> int:
    task = AlgoTask.model_validate(data)
    check_output(task)
    versions = installed_versions()
    algos = {}
    for name in type(task.algos).model_fields:
        component = getattr(task.algos, name)
        if component:
            algos[name] = load_component(component, Algo)
            log.info(f"加载 {name}：{component.package} {component.version}（{component.entry}）")
    context = (
        load_entry(distribution(task.algos.model.package), task.context)
        if task.context
        else context_type(algos["model"])
    )
    if not issubclass(context, ResearchContext):
        raise TypeError("Context 必须继承当前 scheme 的 ResearchContext")
    if resume_run(task, BacktestResult, versions, checkpoints):
        return 0
    strategy = Strategy(context(), params=task.backtest, **algos)
    result = run_backtest(strategy.algos, strategy.ctx, task.backtest, checkpoints=checkpoints)
    save_run(task, result, versions, checkpoints)
    return 0
