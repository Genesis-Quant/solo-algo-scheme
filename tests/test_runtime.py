from datetime import date
from importlib.metadata import distribution

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field, ValidationError

from scheme import Algo, Factor, FactorParams, ResearchContext, StockPool, Universe
from scheme.base import (
    BacktestParameters,
    ControlAlgo,
    ControlParams,
    ControlReportForm,
    ExecutionAlgo,
    ExecutionParams,
    ExecutionReportForm,
    FactorAnalysisParams,
    FactorReportForm,
    ModelAlgo,
    ModelAnalysisParams,
    ModelParams,
    ModelReportForm,
    OptimizeAlgo,
    OptimizeParams,
    OptimizeReportForm,
)
from scheme.base.projects.optimize.default import risk_parity
from scheme.execute.factor.api import prepare_panel
from scheme.execute.packages import load_entry, validate_environment
from scheme.execute.schema import Environment, StrategyTask
from scheme.execute.strategy.assembly import parameter_type
from scheme.manage import main


class Params(BaseModel):
    count: int = 2


class ExampleFactor(Factor[FactorParams]):
    def compute(self, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame(
            {"f": [1.0, 2.0]},
            index=pd.MultiIndex.from_product(
                [[date(2025, 1, 1)], ["000001.XSHE", "600000.XSHG"]], names=["date", "symbol"]
            ),
        )


class ExampleAlgo[C: ResearchContext[float]](Algo[Params, C]):
    pass


def test_parameter_generic_inheritance():
    class Child(ExampleAlgo[ResearchContext[float]]):
        pass

    assert parameter_type(ExampleFactor) is FactorParams
    assert parameter_type(Child) is Params


def test_factor_preserves_panel_values():
    parameters = FactorAnalysisParams(
        start="2025-01-01", end="2025-01-03", columns=["f"], return_periods=[1], groups=2
    )
    panel = ExampleFactor(parameters).compute(parameters.start, parameters.end)
    source = prepare_panel(panel, parameters)
    assert source.f.tolist() == [1.0, 2.0]
    assert source.code.tolist() == ["000001.XSHE", "600000.XSHG"]


def test_factor_rejects_duplicate_panel():
    params = FactorAnalysisParams(start="2025-01-01", end="2025-01-02", columns=["f"])
    panel = ExampleFactor(params).compute(params.start, params.end)
    with pytest.raises(ValueError, match="唯一"):
        prepare_panel(pd.concat([panel, panel]), params)


def test_factor_report_form_build_and_run(monkeypatch):
    form = FactorReportForm(
        start="2025-01-01",
        end="2025-01-03",
        columns=["f"],
        pool=StockPool.CSI300,
        groups=3,
        return_periods=[1, 5],
    )
    params = form.build()
    assert isinstance(params, FactorAnalysisParams)
    assert params.universe.pool == StockPool.CSI300
    assert params.universe.query(params.start, params.end).model_dump()["filters"] == [
        "stock_pool_member"
    ]
    received = {}
    report = object()

    def analyze(factor, analysis, *, settings):
        received.update(factor=factor, analysis=analysis, settings=settings)
        return report

    monkeypatch.setattr("scheme.execute.factor.api.analyze_factors", analyze)
    assert params.run(ExampleFactor) is report
    assert received["analysis"] is params
    assert type(received["factor"].params) is FactorParams
    assert set(received["factor"].params.model_dump()) == {"start", "end", "universe"}
    assert received["factor"].params.universe == params.universe
    with pytest.raises(TypeError, match="Factor 类"):
        params.run(ExampleFactor(params.factor_params))


def test_factor_analysis_accepts_custom_universe():
    universe = Universe(
        codes=["000001.SZ"],
        derivatives={"member": {"type": "DIRECT", "op": "nullary.true", "fields": {}}},
        filters=["member"],
    )
    params = FactorAnalysisParams(
        start="2025-01-01",
        end="2025-01-03",
        columns=["f"],
        universe=universe,
    )
    assert params.factor_params.universe.codes == ["000001.SZ"]
    with pytest.raises(ValidationError):
        FactorReportForm(start="2025-01-01", end="2025-01-03", columns=["f"], pool=StockPool.CUSTOM)


@pytest.mark.parametrize(
    "kind,form_type",
    [
        ("optimize", OptimizeReportForm),
        ("control", ControlReportForm),
        ("execution", ExecutionReportForm),
    ],
)
def test_downstream_form_runs_current_stage(monkeypatch, kind, form_type):
    from scheme.base.projects.control.default import NoControl
    from scheme.base.projects.execution.default import DirectExecution
    from scheme.base.projects.optimize.default import RiskParity

    class State(ResearchContext[float]):
        pass

    class Model(ModelAlgo[ModelParams, State]):
        pass

    classes = {
        "model": Model,
        "optimize": RiskParity,
        "control": NoControl,
        "execution": DirectExecution,
    }
    names = list(classes)
    upstream = names[: names.index(kind)]
    selected = {name: f"{name}_test:{classes[name].__name__}" for name in upstream}
    monkeypatch.setattr(
        "scheme.execute.strategy.components.resolve_algo", lambda name, entry: classes[name]
    )
    captured = {}
    report = object()

    def backtest(algos, ctx, params, *, settings):
        captured.update(algos=algos, ctx=ctx, params=params)
        return report

    monkeypatch.setattr("scheme.execute.strategy.api.run_backtest", backtest)
    form = form_type(start="2025-01-01", end="2025-01-03", cash=300_000, **selected)
    params = type(form.build()).model_validate_json(form.build().model_dump_json())
    assert kind not in form_type.model_fields
    assert params.run(classes[kind]) is report
    assert [type(algo) for algo in captured["algos"]] == list(classes.values())
    assert type(captured["ctx"]) is State
    assert params.config["cash"] == 300_000
    with pytest.raises(TypeError, match="run 需要"):
        params.run(Model)
    with pytest.raises(ValidationError):
        form_type(start="2025-01-01", end="2025-01-03")


def test_upstream_must_be_installed_and_correct_kind(monkeypatch):
    from scheme.execute.strategy.components import resolve_algo

    monkeypatch.setattr("scheme.execute.strategy.components.algo_options", lambda _: {})
    with pytest.raises(ValueError, match="上游未安装"):
        resolve_algo("model", "missing:ModelAlgo")


def test_research_combines_independent_algo_parameters(monkeypatch):
    class ModelSettings(ModelParams):
        n_select: int = Field(default=10, ge=1, title="选股数量")

    class ModelForm(ModelSettings, ModelReportForm):
        pass

    class OptimizeSettings(OptimizeParams):
        gross_exposure: float = Field(default=0.98, gt=0, le=1)

    class OptimizeForm(OptimizeSettings, OptimizeReportForm):
        pass

    class ControlSettings(ControlParams):
        lot_size: int = Field(default=100, ge=1)

    class ControlForm(ControlSettings, ControlReportForm):
        pass

    class ExecutionSettings(ExecutionParams):
        pass

    class ExecutionForm(ExecutionSettings, ExecutionReportForm):
        pass

    class State(ResearchContext[float]):
        pass

    class Model(ModelAlgo[ModelSettings, State]):
        pass

    class Optimize(OptimizeAlgo[OptimizeSettings, State]):
        def on_signal(self) -> None:
            pass

    class Control(ControlAlgo[ControlSettings, State]):
        def on_target(self) -> None:
            pass

    class Execution(ExecutionAlgo[ExecutionSettings, State]):
        def on_orders(self) -> None:
            pass

    classes = dict(model=Model, optimize=Optimize, control=Control, execution=Execution)
    monkeypatch.setattr(
        "scheme.execute.strategy.components.resolve_algo", lambda kind, entry: classes[kind]
    )
    captured = []
    monkeypatch.setattr(
        "scheme.execute.strategy.api.run_backtest",
        lambda algos, ctx, params, **kwargs: captured.extend(algos),
    )
    fields = ExecutionForm.model_json_schema()["properties"]
    assert not {"n_select", "gross_exposure", "lot_size"} & fields.keys()
    assert "n_select" not in OptimizeForm.model_fields
    assert "gross_exposure" not in ControlForm.model_fields
    assert "execution" not in fields
    assert "optimize" not in OptimizeForm.model_fields
    assert "control" not in ControlForm.model_fields
    form = ExecutionForm(
        start="2025-01-01", end="2025-01-03",
        model="model_test:Model", optimize="optimize_test:Optimize", control="control_test:Control",
    )
    params = form.build()
    params = type(params)(
        **params.model_dump(),
        **ModelSettings(n_select=3).model_dump(),
        **OptimizeSettings(gross_exposure=0.7).model_dump(),
        **ControlSettings(lot_size=200).model_dump(),
    )
    assert type(params).__name__ == "ExecutionAnalysisParams"
    params.run(Execution)
    assert captured[0].params.n_select == 3
    assert captured[1].params.gross_exposure == 0.7
    assert captured[2].params.lot_size == 200
    assert captured[3].params.model_dump() == {}
    with pytest.raises(ValidationError):
        ExecutionForm(**{**form.model_dump(), "n_select": 0})


def test_model_form_runs_complete_chain(monkeypatch):
    class Parameters(ModelParams):
        count: int = 1

    class State(ResearchContext[float]):
        count: int = 0

    class Model(ModelAlgo[Parameters, State]):
        pass

    class Form(ModelReportForm):
        count: int = 3

    form = Form(start="2025-01-01", end="2025-01-03", cash=500_000)
    params = ModelAnalysisParams.model_validate_json(form.build().model_dump_json())
    assert params.universe.pool == StockPool.CSI300
    assert params.config["cash"] == 500_000
    captured = {}
    report = object()

    def backtest(algos, ctx, parameters, *, settings):
        captured.update(algos=algos, ctx=ctx, params=parameters)
        return report

    monkeypatch.setattr("scheme.execute.strategy.api.run_backtest", backtest)
    assert params.run(Model) is report
    assert [type(a).__name__ for a in captured["algos"]] == [
        "Model",
        "RiskParity",
        "NoControl",
        "DirectExecution",
    ]
    assert captured["algos"][0].params.count == 3
    assert type(captured["ctx"]) is State
    ctx = State(count=4)
    assert params.run(Model, ctx=ctx) is report
    assert captured["ctx"] is ctx
    with pytest.raises(TypeError, match="ModelAlgo"):
        params.run(ExampleFactor)
    with pytest.raises(ValidationError):
        Form(start="2025-01-01", end="2025-01-03", optimize="missing")
    with pytest.raises(ValidationError):
        Form(start="2025-01-03", end="2025-01-01").build()
    assert Form.model_json_schema()["properties"]["optimize"]["const"] == "risk_parity"


def test_risk_parity_equal_risk_and_fallback():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20000, 2)) * [1.0, 3.0]
    weights, reason = risk_parity(pd.DataFrame(x))
    assert reason is None
    assert weights == pytest.approx([0.75, 0.25], abs=0.015)
    covariance = np.cov(x.T)
    risk = weights * (covariance @ weights)
    assert risk[0] / risk[1] == pytest.approx(1.0, abs=0.01)
    weights, reason = risk_parity(pd.DataFrame(x[:2]))
    assert reason == "历史样本不足"
    assert weights == pytest.approx([0.5, 0.5])
    _, reason = risk_parity(pd.DataFrame(np.zeros((30, 2))))
    assert reason == "协方差退化"


@pytest.mark.parametrize(
    "updates",
    [
        {"end": "2025-01-01"},
        {"batch_days": 0},
        {"config": {"startDate": "2025-01-01"}},
        {"symbols": []},
    ],
)
def test_backtest_rejects_invalid_config(updates):
    with pytest.raises(ValidationError):
        BacktestParameters.model_validate(
            {"start": "2025-01-01", "end": "2025-01-03", "symbols": ["A"], **updates}
        )


def test_entry_must_belong_to_distribution():
    with pytest.raises(ValueError, match="不属于"):
        load_entry(distribution("scheme"), "pathlib:Path")


def test_version_mismatch_before_execution(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname="solo-runtime"\nversion="2.0.0"\n')
    with pytest.raises(ValueError, match="不在任务锁文件"):
        validate_environment(Environment(lockfile=lock))


def test_cli_failure_returns_nonzero(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"kind":"factor"}')
    assert main(["run", "--input", str(source), "--output", str(tmp_path / "report")]) == 1


def test_unknown_input_field_rejected():
    with pytest.raises(ValidationError):
        StrategyTask.model_validate({"kind": "strategy", "typo": True})
