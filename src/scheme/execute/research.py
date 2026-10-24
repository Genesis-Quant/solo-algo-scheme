from importlib.metadata import distribution

from runtime.utils import logger as log

from scheme.base import Algo, ResearchContext
from scheme.execute.checkpoints import Checkpoints
from scheme.execute.packages import installed_versions, load_component, load_entry
from scheme.execute.results import check_output, resume_run, save_run
from scheme.execute.schema import AlgoTask
from scheme.execute.strategy import BacktestResult, run_backtest
from scheme.execute.strategy.assembly import Strategy, context_type


def run(task: AlgoTask, *, checkpoints: Checkpoints | None = None) -> int:
    """当前五种 Algo/策略任务共用的回测报告实现，由各自任务入口调用。"""
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
