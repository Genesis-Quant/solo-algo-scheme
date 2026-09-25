from datetime import date
from importlib.metadata import distribution

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from scheme import Algo, Factor, FactorParams, ResearchContext, StockPool, Universe
from scheme.base import (
    BacktestParameters,
    ControlReportForm,
    ExecutionReportForm,
    FactorAnalysisParams,
    FactorReportForm,
    ModelReportForm,
    OptimizeReportForm,
    ReportForm,
)
from scheme.base.projects.optimize.default import risk_parity
from scheme.execute.factor.api import prepare_panel
from scheme.execute.packages import load_entry, validate_environment
from scheme.execute.schema import BacktestTask, Environment
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
    "form_type", [ModelReportForm, OptimizeReportForm, ControlReportForm, ExecutionReportForm]
)
def test_strategy_project_forms_require_their_own_build(form_type):
    assert form_type.model_fields == {}
    with pytest.raises(TypeError, match="abstract"):
        form_type()

    class ProjectForm(form_type[Params]):
        count: int = 2

        def build(self) -> Params:
            return Params(count=self.count)

    form = ProjectForm(count=5)
    assert isinstance(form, ReportForm)
    assert isinstance(form, form_type[Params])
    assert form_type[Params].__pydantic_generic_metadata__["args"] == (Params,)
    assert form.build() == Params(count=5)


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
        BacktestTask.model_validate({"kind": "backtest", "typo": True})
