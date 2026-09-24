from importlib.metadata import distribution
from typing import Any

from scheme import Algo, ResearchContext, Strategy
from scheme.apps.backtest import run_backtest
from scheme.apps.backtest.schema import BacktestTask
from scheme.utils.manage import check_output, save_run
from scheme.utils.packages import load_component, load_entry, validate_environment


def run(data: dict[str, Any], *, input_sha256: str) -> int:
    task = BacktestTask.model_validate(data)
    check_output(task)
    versions = validate_environment(task.environment)
    context_type = load_entry(distribution(task.algos.model.package), task.context)
    if not issubclass(context_type, ResearchContext):
        raise TypeError("Context 必须继承当前 scheme 的 ResearchContext")
    algos = {}
    for name in type(task.algos).model_fields:
        component = getattr(task.algos, name)
        if component:
            algos[name] = load_component(component, Algo)
    strategy = Strategy(context_type(), params=task.backtest, **algos)
    result = run_backtest(strategy.algos, strategy.ctx, task.backtest)
    save_run(task, result, versions, input_sha256=input_sha256)
    return 0
