from importlib.metadata import distribution

from scheme.base import Algo, ResearchContext
from scheme.execute.packages import load_component, load_entry, validate_environment
from scheme.execute.results import check_output, save_run
from scheme.execute.schema import AlgoTask
from scheme.execute.strategy import run_backtest
from scheme.execute.strategy.assembly import Strategy, context_type


def run(task: AlgoTask, *, input_sha256: str) -> int:
    """当前五种 Algo/策略任务共用的回测报告实现，由各自任务入口调用。"""
    check_output(task)
    versions = validate_environment(task.environment)
    algos = {}
    for name in type(task.algos).model_fields:
        component = getattr(task.algos, name)
        if component:
            algos[name] = load_component(component, Algo)
    context = (
        load_entry(distribution(task.algos.model.package), task.context)
        if task.context
        else context_type(algos["model"])
    )
    if not issubclass(context, ResearchContext):
        raise TypeError("Context 必须继承当前 scheme 的 ResearchContext")
    strategy = Strategy(context(), params=task.backtest, **algos)
    result = run_backtest(strategy.algos, strategy.ctx, task.backtest)
    save_run(task, result, versions, input_sha256=input_sha256)
    return 0
