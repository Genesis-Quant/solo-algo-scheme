from contextlib import nullcontext
from copy import deepcopy
from datetime import date
from importlib.metadata import distribution
from types import SimpleNamespace
from typing import assert_type
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, Field, ValidationError

from scheme import Factor, FactorParams, ResearchContext, StockPool, Strategy, Universe
from scheme.base import (
    ControlAlgo,
    ControlAnalysisParams,
    ControlParams,
    ControlReportForm,
    ExecutionAlgo,
    ExecutionAnalysisParams,
    ExecutionParams,
    ExecutionReportForm,
    FactorAnalysisParams,
    FactorReportForm,
    ModelAlgo,
    ModelAnalysisParams,
    ModelParams,
    ModelReportForm,
    OptimizeAlgo,
    OptimizeAnalysisParams,
    OptimizeParams,
    OptimizeReportForm,
    StrategyAnalysisParams,
    StrategyParams,
)
from scheme.base.projects.optimize.default import risk_parity
from scheme.config import DolphinSettings
from scheme.data.dolphindb.universe import MEMBERS
from scheme.execute.factor.api import analyze_factors, prepare_panel
from scheme.execute.factor.result import FactorAnalysisResult
from scheme.execute.packages import load_entry
from scheme.execute.schema import FactorTask, StrategyTask
from scheme.execute.strategy.result import BacktestResult
from scheme.manage import main

DATES = {"start": "2025-01-01", "end": "2025-01-03"}


class ExampleFactor(Factor[FactorParams]):
    def compute(self, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame(
            {"f": [1.0, 2.0]},
            index=pd.MultiIndex.from_product(
                [[date(2025, 1, 1)], ["000001.XSHE", "600000.XSHG"]],
                names=["date", "symbol"],
            ),
        )


class State(ResearchContext[float]):
    count: int = 0


class OtherState(ResearchContext[str]):
    pass


def empty_backtest(parameters=None):
    return BacktestResult(
        **{name: pd.DataFrame() for name in BacktestResult.filenames},
        parameters=None if parameters is None else parameters.model_copy(deep=True),
    )


def empty_factor_report(parameters):
    return FactorAnalysisResult(
        **{name: pd.DataFrame() for name in FactorAnalysisResult.filenames},
        parameters=parameters.model_copy(deep=True),
    )


@pytest.fixture
def backtest_spy(monkeypatch):
    calls = []
    report = empty_backtest()

    def backtest(algos, ctx, parameters, *, settings):
        calls.append(dict(algos=algos, ctx=ctx, parameters=parameters, settings=settings))
        report.parameters = parameters.model_copy(deep=True)
        return report

    monkeypatch.setattr("scheme.execute.strategy.api.run_backtest", backtest)
    return calls, report


@pytest.fixture
def settings():
    return DolphinSettings(host="unused.invalid", username="offline", password="unused")


def test_factor_preserves_panel_values():
    parameters = FactorAnalysisParams(**DATES, columns=["f"], max_lag=1, groups=2)
    panel = ExampleFactor(FactorParams()).compute(parameters.start, parameters.end)
    source = prepare_panel(panel, parameters)
    assert source.f.tolist() == [1.0, 2.0]
    assert source.code.tolist() == ["000001.XSHE", "600000.XSHG"]


def test_factor_rejects_duplicate_panel():
    params = FactorAnalysisParams(**DATES, columns=["f"])
    panel = ExampleFactor(FactorParams()).compute(params.start, params.end)
    with pytest.raises(ValueError, match="唯一"):
        prepare_panel(pd.concat([panel, panel]), params)


@pytest.mark.parametrize("day", ["2024-12-31", "2025-01-04"])
def test_factor_panel_rejects_dates_outside_closed_interval(day):
    params = FactorAnalysisParams(**DATES, columns=["f"])
    panel = pd.DataFrame(
        {"f": [1.0]},
        index=pd.MultiIndex.from_tuples([(date.fromisoformat(day), "000001.SZ")],
                                        names=["date", "symbol"]),
    )
    with pytest.raises(ValueError, match="因子面板为空"):
        prepare_panel(panel, params)
    source = prepare_panel(pd.concat([
        panel,
        ExampleFactor(FactorParams()).compute(params.start, params.end),
    ]), params)
    assert source.time.tolist() == [pd.Timestamp(DATES["start"])] * 2
    assert source.f.tolist() == [1.0, 2.0]


@pytest.mark.parametrize("start,end", [
    ("2025-01-01", "2025-01-03"), ("2025-01-03", "2025-01-03"),
])
def test_factor_panel_includes_both_closed_interval_boundaries(start, end):
    params = FactorAnalysisParams(start=start, end=end, columns=["f"])
    dates = pd.to_datetime(["2024-12-31", "2025-01-01", "2025-01-03", "2025-01-04"])
    panel = pd.DataFrame(
        {"f": [0.0, 1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_product([dates, ["000001.XSHE"]], names=["date", "symbol"]),
    )
    source = prepare_panel(panel, params)
    expected = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    assert source.time.tolist() == expected.tolist()


def test_factor_panel_converts_numeric_values_without_coercing_invalid_input():
    params = FactorAnalysisParams(**DATES, columns=["f"])
    panel = ExampleFactor(FactorParams()).compute(params.start, params.end)
    panel["f"] = ["1.25", "2"]
    assert prepare_panel(panel, params).f.tolist() == [1.25, 2.0]
    panel.loc[:, "f"] = "not-a-number"
    with pytest.raises(ValueError, match="Unable to parse"):
        prepare_panel(panel, params)


@pytest.mark.parametrize("max_lag", [1, 2, 5])
@pytest.mark.parametrize("period", [1, 5])
@pytest.mark.parametrize("price_column", ["open", "close", "open_hfq", "close_hfq"])
def test_factor_returns_are_read_only_until_end(monkeypatch, settings, max_lag, period, price_column):
    requests = []

    class Stop(Exception):
        pass

    def capture(request, *, session):
        requests.append(request)
        raise Stop

    monkeypatch.setattr("scheme.execute.factor.api.query", capture)
    monkeypatch.setattr("scheme.execute.factor.api.create_session", lambda _: nullcontext(object()))
    params = FactorAnalysisParams(
        **DATES, columns=["f"], period=period, max_lag=max_lag, price_column=price_column,
    )
    with pytest.raises(Stop):
        analyze_factors(ExampleFactor(FactorParams()), params, settings=settings)
    assert (requests[0]["start_date"], requests[0]["end_date"]) == ("2025-01-01", "2025-01-03")
    assert list(requests[0]["derivatives"]) == params.return_columns
    from runtime.apps.query import FactorQuery

    validated = FactorQuery.model_validate(requests[0])
    assert set(validated.source_factors()) == {price_column}


@pytest.mark.parametrize("start", ["2025-01-01", "2025-01-03"])
def test_factor_analysis_keeps_closed_end_in_source_query_and_statistics(monkeypatch, settings, start):
    from scheme.execute.factor import api

    params = FactorAnalysisParams(start=start, end="2025-01-03", columns=["f"])
    panel = pd.DataFrame(
        {"f": [1.0, 2.0]},
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2025-01-01", "2025-01-03"]), ["000001.XSHE"]],
            names=["date", "symbol"],
        ),
    )
    factor = Mock(spec=ExampleFactor)
    factor.compute.return_value = panel
    session = SimpleNamespace(run=Mock(return_value=pd.DataFrame()), upload=Mock())
    request = Mock(return_value=nullcontext(SimpleNamespace(data_ref="offlineReturns")))
    monkeypatch.setattr(api, "create_session", lambda _: nullcontext(session))
    monkeypatch.setattr(api, "query", request)

    analyze_factors(factor, params, settings=settings)
    factor.compute.assert_called_once_with(params.start, params.end)
    assert request.call_args.args[0]["start_date"] == start
    assert request.call_args.args[0]["end_date"] == "2025-01-03"
    uploaded = session.upload.call_args.args[0]
    assert uploaded["soloStart"] == np.datetime64(params.start, "ms")
    assert uploaded["soloEnd"] == np.datetime64("2025-01-04", "ms")
    assert uploaded["soloFactorSource"].time.max() == pd.Timestamp(params.end)
    assert len(uploaded["soloFactorSource"]) == (2 if start == "2025-01-01" else 1)
    assert any("soloStart, soloEnd" in call.args[0] for call in session.run.call_args_list)


def test_factor_report_form_build_and_run(monkeypatch, settings):
    form = FactorReportForm(
        **DATES, columns=["f"], pool=StockPool.CSI300, groups=3,
        period=5, price_column="close_hfq", max_lag=3,
    )
    params = form.build()
    assert_type(params, FactorAnalysisParams)
    assert type(params) is FactorAnalysisParams
    assert (params.period, params.price_column, params.max_lag) == (5, "close_hfq", 3)
    assert params.return_columns == ["P5L1", "P5L2", "P5L3"]
    assert (params.start, params.end) == (form.start, form.end)
    assert FactorAnalysisParams.model_validate_json(params.model_dump_json()) == params
    assert params.universe == Universe(pool=StockPool.CSI300)
    report = empty_factor_report(params)
    analyze = Mock(return_value=report)
    monkeypatch.setattr("scheme.execute.factor.api.analyze_factors", analyze)

    result = params.run(ExampleFactor, settings=settings)
    assert_type(result, FactorAnalysisResult[FactorAnalysisParams])
    assert result is report
    factor, analysis = analyze.call_args.args
    assert analysis is params
    assert analyze.call_args.kwargs == {"settings": settings}
    assert type(factor.params) is FactorParams
    assert set(factor.params.model_dump()) == {"universe"}
    assert not hasattr(factor, "universe")
    assert factor.params.universe == params.universe
    assert type(result.parameters) is FactorAnalysisParams
    assert result.parameters == params
    assert not hasattr(params, "factor_params")
    with pytest.raises(TypeError, match="Factor 类"):
        params.run(ExampleFactor(FactorParams()))
    with pytest.raises(TypeError, match="Factor 类"):
        params.run(ModelAlgo)
    assert analyze.call_count == 1


INDEX_MEMBER = {
    "type": "DIRECT", "op": "binary.gt", "fields": {"left": "weight_000300SH", "right": 0}, "params": {},
}
ALL_MARKET_MEMBER = {"type": "DIRECT", "op": "nullary.true", "fields": {}, "params": {}}


def test_universe_offers_only_arena_stock_pools():
    assert [pool.value for pool in StockPool] == ["全市场", "上证 50", "沪深 300", "中证 500", "中证 1000"]
    assert Universe() == Universe(pool=StockPool.ALL)
    assert FactorAnalysisParams(**DATES, columns=["f"]).universe == Universe()
    for value in ({"pool": "项目股票池"}, {"pool": StockPool.CSI300, "lookback": "P5D"},
                  {"pool": StockPool.CSI300, "filters": ["member"]}):
        with pytest.raises(ValidationError):
            Universe.model_validate(value)
    with pytest.raises(ValidationError):
        FactorReportForm(**DATES, columns=["f"], pool="项目股票池")


def test_member_is_a_class_attribute_not_a_parameter():
    universe = Universe(pool=StockPool.CSI300)
    assert Universe.member == universe.member == "stock_pool_member"
    assert "member" not in Universe.model_fields
    assert universe.model_dump(mode="json") == {"pool": "沪深 300"}
    with pytest.raises(ValidationError):
        Universe.model_validate({"pool": StockPool.CSI300, "member": "other"})
    with pytest.raises(AttributeError):
        universe.member = "other"


def test_index_pool_codes_query_is_arena_first_stage():
    codes_query = Universe(pool=StockPool.CSI300).codes_query(date(2025, 1, 1), date(2025, 1, 3))
    assert codes_query.model_dump(mode="json") == {
        "start_date": "2025-01-01", "end_date": "2025-01-02", "lookback": "PT0S",
        "codes": [], "factors": [], "derivatives": {"stock_pool_member": INDEX_MEMBER},
        "filters": ["stock_pool_member"],
    }
    weights = {
        StockPool.SSE50: "weight_000016SH", StockPool.CSI300: "weight_000300SH",
        StockPool.CSI500: "weight_000905SH", StockPool.CSI1000: "weight_000852SH",
    }
    for pool, weight in weights.items():
        query = Universe(pool=pool).codes_query(date(2025, 1, 1), date(2025, 1, 3))
        assert query.model_dump(mode="json")["derivatives"]["stock_pool_member"] == {
            **INDEX_MEMBER, "fields": {"left": weight, "right": 0},
        }
    assert Universe().codes_query(date(2025, 1, 1), date(2025, 1, 3)) is None


@pytest.mark.parametrize("pool", list(StockPool))
def test_every_stock_pool_injects_its_mapped_member_definition(pool):
    query = Universe(pool=pool).dataset_query(date(2025, 1, 1), date(2025, 1, 3), factors=["close"])
    assert query.derivatives[Universe.member].model_dump(mode="json") == MEMBERS[pool]
    assert query.filters == []


@pytest.mark.parametrize("pool,member", [
    (StockPool.CSI300, INDEX_MEMBER), (StockPool.ALL, ALL_MARKET_MEMBER),
])
def test_dataset_query_exposes_member_column_without_filtering(pool, member):
    universe = Universe(pool=pool)
    derivatives = {
        "rank": {
            "type": "CS", "op": "unary.rank_pct", "fields": {"col": "close"},
            "params": {}, "on": Universe.member,
        },
        "valid": {"type": "DIRECT", "op": "binary.gt", "fields": {"left": "close", "right": 0}},
    }
    query = universe.dataset_query(
        date(2025, 1, 1), date(2025, 1, 3),
        factors=["close"], derivatives=derivatives, filters=["valid"],
    ).model_dump(mode="json")
    assert (query["start_date"], query["end_date"], query["codes"]) == ("2025-01-01", "2025-01-02", [])
    assert list(query["derivatives"]) == ["stock_pool_member", "rank", "valid"]
    assert query["derivatives"]["stock_pool_member"] == member
    assert query["derivatives"]["rank"]["on"] == "stock_pool_member"
    assert query["filters"] == ["valid"]
    kept = universe.dataset_query(
        date(2025, 1, 1), date(2025, 1, 3), factors=["close"], filters=["valid", Universe.member],
        derivatives={"valid": derivatives["valid"]},
    )
    assert kept.filters == ["valid", "stock_pool_member"]
    for authored in ({"factors": [Universe.member]},
                     {"derivatives": {Universe.member: ALL_MARKET_MEMBER}}):
        with pytest.raises(ValueError, match="stock_pool_member"):
            universe.dataset_query(date(2025, 1, 1), date(2025, 1, 3), **authored)


@pytest.mark.parametrize(
    "kind,form_type,analysis_type",
    [
        ("optimize", OptimizeReportForm, OptimizeAnalysisParams),
        ("control", ControlReportForm, ControlAnalysisParams),
        ("execution", ExecutionReportForm, ExecutionAnalysisParams),
    ],
)
def test_downstream_form_runs_current_stage(
    monkeypatch, backtest_spy, settings, kind, form_type, analysis_type,
):
    from scheme.base.projects.control.default import NoControl
    from scheme.base.projects.execution.default import DirectExecution
    from scheme.base.projects.optimize.default import RiskParity

    class Model(ModelAlgo[ModelParams, State]):
        pass

    class Analysis(analysis_type):
        report_label: str = "current-stage"

    classes = dict(model=Model, optimize=RiskParity, control=NoControl, execution=DirectExecution)
    upstream = list(classes)[:list(classes).index(kind)]
    selected = {name: f"{name}_test:{classes[name].__name__}" for name in upstream}
    resolve = Mock(side_effect=lambda name, entry: classes[name])
    monkeypatch.setattr("scheme.execute.strategy.components.resolve_algo", resolve)
    form = form_type(**DATES, cash=300_000, **selected)
    built = form.build()
    assert type(built) is analysis_type
    params = Analysis.model_validate_json(built.model_dump_json())
    calls, report = backtest_spy

    assert params.run(classes[kind], settings=settings) is report
    assert [type(algo) for algo in calls[-1]["algos"]] == list(classes.values())
    assert type(calls[-1]["ctx"]) is State
    assert calls[-1]["settings"] is settings
    assert type(calls[-1]["parameters"]) is StrategyAnalysisParams
    assert type(report.parameters) is Analysis
    assert report.parameters == params
    assert report.parameters is not params
    assert report.parameters.config is not params.config
    assert params.config["cash"] == 300_000
    assert [call.args for call in resolve.call_args_list] == [
        (name, selected[name]) for name in upstream
    ]
    explicit = State(count=4)
    assert params.run(classes[kind], ctx=explicit) is report
    assert calls[-1]["ctx"] is explicit
    with pytest.raises(TypeError, match="run 需要"):
        params.run(Model)
    assert len(calls) == 2
    with pytest.raises(ValidationError):
        form_type(**DATES)


def test_upstream_must_be_installed_and_correct_kind(monkeypatch):
    from scheme.execute.strategy.components import resolve_algo

    monkeypatch.setattr("scheme.execute.strategy.components.algo_options", lambda _: {})
    with pytest.raises(ValueError, match="上游未安装"):
        resolve_algo("model", "missing:ModelAlgo")
    monkeypatch.setattr(
        "scheme.execute.strategy.components.algo_options", lambda _: {"fixture:Algo": "1.0"}
    )
    monkeypatch.setattr(
        "scheme.execute.strategy.components.importlib.import_module",
        lambda _: type("Module", (), {"Algo": ExampleFactor}),
    )
    with pytest.raises(TypeError, match="ModelAlgo"):
        resolve_algo("model", "fixture:Algo")


@pytest.mark.parametrize(
    "version,expected",
    [
        ("1.0.0", {"model_old:ModelAlgo"}),
        ("1.0.7", {"model_old:ModelAlgo"}),
        ("1.1.0", {
            "model_minimum:ModelAlgo", "model_patch:ModelAlgo", "model_shorthand:ModelAlgo",
            "model_patch_upper:ModelAlgo", "model_cross_minor_lower:ModelAlgo", "model_exact:ModelAlgo",
            "model_excluded:ModelAlgo", "model_conditional:ModelAlgo", "model_extra:ModelAlgo",
            "model_duplicate_bound:ModelAlgo", "model_compatible:ModelAlgo", "model_wildcard:ModelAlgo",
        }),
        ("1.1.7", {
            "model_minimum:ModelAlgo", "model_patch:ModelAlgo", "model_shorthand:ModelAlgo",
            "model_patch_upper:ModelAlgo", "model_cross_minor_lower:ModelAlgo",
            "model_excluded:ModelAlgo", "model_conditional:ModelAlgo", "model_extra:ModelAlgo",
            "model_duplicate_bound:ModelAlgo", "model_compatible:ModelAlgo", "model_wildcard:ModelAlgo",
        }),
        ("1.2.0", {"model_new:ModelAlgo"}),
        ("2.3.0", {"model_other_major:ModelAlgo", "model_other_major_minimum:ModelAlgo"}),
        ("2.3.7", {"model_other_major:ModelAlgo", "model_other_major_minimum:ModelAlgo"}),
    ],
)
def test_algo_options_match_formal_scheme_compatibility(monkeypatch, version, expected):
    from scheme.execute.strategy import components

    candidates = {
        "old": ("1.0.9", ["scheme>=1.0,<1.1"]),
        "minimum": ("1.1.0", ["scheme>=1.1.0,<1.2.0"]),
        "patch": ("1.1.7", ["scheme>=1.1.0,<1.2.0"]),
        "shorthand": ("1.1", ["pydantic>=2,<3", "ScHeMe (<1.2, >=1.1)"]),
        "compatible": ("1.1.0", ["scheme~=1.1.0"]),
        "wildcard": ("1.1.0", ["scheme==1.1.*"]),
        "new": ("1.2.0", ["scheme>=1.2.0,<1.3.0"]),
        "other_major": ("2.3.7", ["scheme>=2.3.0,<2.4.0"]),
        "other_major_minimum": ("2.3.0", ["scheme>=2.3.0,<2.4.0"]),
        "package_older_minor": ("1.0.7", ["scheme>=1.1.0,<1.2.0"]),
        "package_newer_minor": ("1.2.7", ["scheme>=1.1.0,<1.2.0"]),
        "package_other_major": ("2.1.7", ["scheme>=1.1.0,<1.2.0"]),
        "old_range_old_package": ("1.0.7", ["scheme>=1.0.0,<2.0.0"]),
        "old_range_new_package": ("1.1.7", ["scheme>=1.1.0,<2.0.0"]),
        "patch_floor": ("1.1.7", ["scheme>=1.1.1,<1.2.0"]),
        "project_patch_floor": ("1.1.7", ["scheme>=1.1.7,<1.2.0"]),
        "patch_upper": ("1.1.7", ["scheme>=1.1.0,<1.1.8"]),
        "cross_minor_lower": ("1.1.0", ["scheme>=1.0.0,<1.2.0"]),
        "cross_minor_upper": ("1.1.0", ["scheme>=1.1.0,<1.3.0"]),
        "exact": ("1.1.0", ["scheme==1.1.0"]),
        "unbounded": ("1.1.0", ["scheme>=1.1.0"]),
        "excluded": ("1.1.0", ["scheme>=1.1.0,<1.2.0,!=1.1.2"]),
        "url": ("1.1.0", ["scheme @ https://example.invalid/scheme.whl"]),
        "conditional": ("1.1.0", ['scheme>=1.1.0,<1.2.0; python_version >= "3.12"']),
        "extra": ("1.1.0", ["scheme[optional]>=1.1.0,<1.2.0"]),
        "duplicate": ("1.1.0", ["scheme>=1.1.0,<1.2.0", "scheme>=1.1,<1.2"]),
        "duplicate_bound": ("1.1.0", ["scheme>=1.1.0,>=1.1.0,<1.2.0"]),
        "malformed_requirement": ("1.1.0", ["scheme>=not-a-version,<1.2.0"]),
    }
    for index, invalid_version in enumerate([
        "not-a-version", "1!1.1.0", "1.1.0rc1", "1.1.0.dev1",
        "1.1.0.post1", "1.1.0+local", "1.1.0.0", "1.1.0.1",
    ]):
        candidates[f"invalid_version_{index}"] = (invalid_version, ["scheme>=1.1.0,<1.2.0"])
    distributions = [
        SimpleNamespace(metadata={"Name": f"model-{name}"}, version=package_version, requires=values)
        for name, (package_version, values) in candidates.items()
    ]
    monkeypatch.setattr(components.metadata, "version", lambda _: version)
    monkeypatch.setattr(components.metadata, "distributions", lambda: distributions)
    options = components.algo_options("model")
    assert set(options) == expected
    assert list(options) == sorted(options)
    for name in ("minimum", "patch"):
        entry = f"model_{name}:ModelAlgo"
        if entry in options:
            assert options[entry] == f"model-{name} · {candidates[name][0]}"


@pytest.mark.parametrize("version", [
    "not-a-version", "1!1.1.0", "1.1.0rc1", "1.1.0.dev1",
    "1.1.0.post1", "1.1.0+local", "1.1.0.0", "1.1.0.1",
])
def test_algo_options_reject_non_release_scheme_versions(monkeypatch, version):
    from scheme.execute.strategy import components

    dist = SimpleNamespace(
        metadata={"Name": "model-test"}, version="1.1.0", requires=["scheme>=1.1.0,<1.2.0"],
    )
    monkeypatch.setattr(components.metadata, "version", lambda _: version)
    monkeypatch.setattr(components.metadata, "distributions", lambda: [dist])
    if version == "not-a-version":
        with pytest.raises(ValueError):
            components.algo_options("model")
    else:
        assert components.algo_options("model") == {}


def test_research_combines_only_declared_independent_algo_parameters(monkeypatch, backtest_spy):
    class ModelSettings(ModelParams):
        n_select: int = Field(ge=1)

    class OptimizeSettings(OptimizeParams):
        gross_exposure: float = Field(gt=0, le=1)

    class ControlSettings(ControlParams):
        lot_size: int = Field(ge=1)

    class ExecutionSettings(ExecutionParams):
        max_orders: int = Field(ge=1)

    class Analysis(ExecutionSettings, ExecutionAnalysisParams):
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
    params = Analysis(
        **DATES, model="model_test:Model", optimize="optimize_test:Optimize",
        control="control_test:Control", n_select=3, gross_exposure=0.7,
        lot_size=200, max_orders=5, analysis_note="not an algorithm parameter",
    )
    params = Analysis.model_validate_json(params.model_dump_json())
    calls, report = backtest_spy
    assert params.run(Execution) is report
    assert [type(algo.params) for algo in calls[-1]["algos"]] == [
        ModelSettings, OptimizeSettings, ControlSettings, ExecutionSettings,
    ]
    assert [algo.params.model_dump() for algo in calls[-1]["algos"]] == [
        {"n_select": 3}, {"gross_exposure": 0.7}, {"lot_size": 200}, {"max_orders": 5},
    ]
    assert type(report.parameters) is Analysis
    assert report.parameters == params
    for key, invalid in (("n_select", 0), ("gross_exposure", 2), ("lot_size", 0)):
        invalid_params = Analysis.model_validate({**params.model_dump(), key: invalid})
        with pytest.raises(ValidationError, match=key):
            invalid_params.run(Execution)
    for key in ("n_select", "gross_exposure", "lot_size"):
        invalid_params = Analysis.model_validate(params.model_dump(exclude={key}))
        with pytest.raises(ValidationError, match=key):
            invalid_params.run(Execution)
    assert len(calls) == 1


def test_model_form_runs_complete_chain(backtest_spy, settings):
    class Parameters(ModelParams):
        count: int = Field(ge=1)

    class Analysis(Parameters, ModelAnalysisParams):
        pass

    class Model(ModelAlgo[Parameters, State]):
        pass

    built = ModelReportForm(**DATES, cash=500_000).build()
    assert type(built) is ModelAnalysisParams
    params = Analysis.model_validate({**built.model_dump(), "count": 3})
    params = Analysis.model_validate_json(params.model_dump_json())
    calls, report = backtest_spy
    result = params.run(Model, settings=settings)
    assert_type(result, BacktestResult[Analysis])
    assert result is report
    assert [type(algo).__name__ for algo in calls[-1]["algos"]] == [
        "Model", "RiskParity", "NoControl", "DirectExecution",
    ]
    assert type(calls[-1]["algos"][0].params) is Parameters
    assert calls[-1]["algos"][0].params.model_dump() == {"count": 3}
    assert type(calls[-1]["ctx"]) is State
    assert calls[-1]["settings"] is settings
    assert type(report.parameters) is Analysis
    assert report.parameters == params
    assert report.parameters is not params
    assert params.universe.pool == StockPool.CSI300
    assert params.config["cash"] == 500_000
    ctx = State(count=4)
    assert params.run(Model, ctx=ctx) is report
    assert calls[-1]["ctx"] is ctx
    with pytest.raises(TypeError, match="ModelAlgo"):
        params.run(ExampleFactor)
    assert len(calls) == 2
    params.config["cash"] = 1
    assert report.parameters.config["cash"] == 500_000
    with pytest.raises(ValidationError):
        ModelReportForm(**DATES, optimize="missing")


def test_strategy_analysis_runs_assembled_strategy(backtest_spy, settings):
    class Parameters(ModelParams):
        count: int = Field(ge=1)

    class Model(ModelAlgo[Parameters, State]):
        pass

    class Runtime(StrategyParams):
        count: int = Field(ge=1)

    class Analysis(Runtime, StrategyAnalysisParams):
        report_label: str = "selection study"

    runtime = Runtime(**DATES, count=7)
    state = State()
    strategy = Strategy(state, params=runtime, model=Model)
    assert_type(strategy, Strategy[Runtime, State])
    assert_type(strategy.params, Runtime)
    assert strategy.params is runtime
    params = Analysis(**runtime.model_dump(), config={"cash": 42_000}, batch_days=2)
    calls, report = backtest_spy
    result = params.run(strategy, settings=settings)
    assert_type(result, BacktestResult[Analysis])
    assert result is report
    assert calls[-1]["algos"] is strategy.algos
    assert calls[-1]["ctx"] is state
    assert calls[-1]["parameters"] is params
    assert calls[-1]["settings"] is settings
    assert type(report.parameters) is Analysis
    assert report.parameters == params
    assert report.parameters.report_label == "selection study"
    assert strategy.algos[0].params.model_dump() == {"count": 7}
    with pytest.raises(TypeError, match="已组装的 Strategy"):
        params.run(Model)
    with pytest.raises(TypeError, match="已组装的 Strategy"):
        params.run(strategy.algos[0])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"start": "2025-01-02"},
        {"end": "2025-01-04"},
        {"universe": {"pool": StockPool.CSI300}},
        {"count": 8},
        {"undeclared_runtime": "different"},
    ],
)
def test_strategy_analysis_rejects_mismatched_runtime(backtest_spy, updates):
    class Model(ModelAlgo[ModelParams, State]):
        pass

    runtime = StrategyParams(**DATES, count=7)
    strategy = Strategy(State(), params=runtime, model=Model)
    params = StrategyAnalysisParams.model_validate({**runtime.model_dump(), **updates})
    with pytest.raises(ValueError, match="运行参数一致"):
        params.run(strategy)
    assert backtest_spy[0] == []


@pytest.mark.parametrize("runtime_values", [{}, {"count": 7}])
def test_strategy_analysis_does_not_exclude_declared_algorithm_fields(
    backtest_spy, runtime_values,
):
    class Parameters(ModelParams):
        count: int = 7

    class Model(ModelAlgo[Parameters, State]):
        pass

    class Analysis(StrategyAnalysisParams):
        count: int
        report_label: str = "selection study"

    runtime = StrategyParams(**DATES, **runtime_values)
    strategy = Strategy(State(), params=runtime, model=Model)
    params = Analysis(**DATES, count=8)
    with pytest.raises(ValueError, match="运行参数一致"):
        params.run(strategy)
    assert backtest_spy[0] == []


@pytest.mark.parametrize("runtime_values", [{}, {"count": "7"}, {"count": 7}])
def test_strategy_analysis_compares_normalized_algorithm_defaults(backtest_spy, runtime_values):
    class Parameters(ModelParams):
        count: int = 7

    class Model(ModelAlgo[Parameters, State]):
        pass

    class Analysis(StrategyAnalysisParams):
        count: int = 7
        report_label: str = "selection study"

    strategy = Strategy(State(), params=StrategyParams(**DATES, **runtime_values), model=Model)
    result = Analysis(**DATES).run(strategy)
    assert result is backtest_spy[1]
    assert result.parameters.count == strategy.algos[0].params.count == 7


@pytest.mark.parametrize("changed", ["runtime", "algo"])
def test_strategy_analysis_rejects_mutation_after_assembly(backtest_spy, changed):
    class Parameters(ModelParams):
        count: int

    class Model(ModelAlgo[Parameters, State]):
        pass

    runtime = StrategyParams(**DATES, count=7)
    strategy = Strategy(State(), params=runtime, model=Model)
    if changed == "runtime":
        runtime.count = 8
    else:
        strategy.algos[0].params.count = 8
    analysis = StrategyAnalysisParams(**dict(runtime))
    with pytest.raises(ValueError, match="运行参数一致"):
        analysis.run(strategy)
    assert backtest_spy[0] == []


def test_strategy_analysis_preserves_unconsumed_runtime_fields(backtest_spy):
    class Model(ModelAlgo[ModelParams, State]):
        pass

    class Analysis(StrategyAnalysisParams):
        note: str

    strategy = Strategy(State(), params=StrategyParams(**DATES, note="same"), model=Model)
    assert Analysis(**DATES, note="same").run(strategy) is backtest_spy[1]
    with pytest.raises(ValueError, match="运行参数一致"):
        Analysis(**DATES, note="different").run(strategy)
    assert len(backtest_spy[0]) == 1


def test_strategy_analysis_checks_algorithm_fields_named_like_analysis(backtest_spy):
    class Parameters(ModelParams):
        batch_days: int

    class Model(ModelAlgo[Parameters, State]):
        pass

    strategy = Strategy(State(), params=StrategyParams(**DATES, batch_days=1), model=Model)
    with pytest.raises(ValueError, match="运行参数一致"):
        StrategyAnalysisParams(**DATES, batch_days=2).run(strategy)
    assert backtest_spy[0] == []
    assert StrategyAnalysisParams(**DATES, batch_days=1).run(strategy) is backtest_spy[1]


def test_strategy_analysis_normalizes_runtime_aliases(backtest_spy):
    class Parameters(ModelParams):
        count: int = Field(alias="n_select")

    class Model(ModelAlgo[Parameters, State]):
        pass

    class Analysis(StrategyAnalysisParams):
        count: int

    strategy = Strategy(State(), params=StrategyParams(**DATES, n_select="7"), model=Model)
    assert Analysis(**DATES, count=7).run(strategy) is backtest_spy[1]
    with pytest.raises(ValueError, match="运行参数一致"):
        Analysis(**DATES, count=8).run(strategy)
    assert len(backtest_spy[0]) == 1


def test_strategy_analysis_normalizes_unconsumed_runtime_alias(backtest_spy):
    class Runtime(StrategyParams):
        note: str = Field(alias="label")

    class Model(ModelAlgo[ModelParams, State]):
        pass

    strategy = Strategy(State(), params=Runtime(**DATES, label="same"), model=Model)
    assert StrategyAnalysisParams(**DATES, label="same").run(strategy) is backtest_spy[1]
    with pytest.raises(ValueError, match="运行参数一致"):
        StrategyAnalysisParams(**DATES, label="different").run(strategy)
    assert len(backtest_spy[0]) == 1


def test_stage_run_preserves_canonical_alias_choice_priority(backtest_spy):
    class Parameters(ModelParams):
        count: int = Field(validation_alias=AliasChoices("n_select", "selection"), gt=0)

    class Analysis(Parameters, ModelAnalysisParams):
        pass

    class Model(ModelAlgo[Parameters, State]):
        pass

    params = Analysis(**DATES, n_select=17, selection=19)
    assert params.count == 17
    result = params.run(Model)
    assert backtest_spy[0][0]["algos"][0].params.count == result.parameters.count == 17
    params.count = 0
    with pytest.raises(ValidationError, match="greater_than"):
        params.run(Model)
    assert len(backtest_spy[0]) == 1


def test_stage_run_preserves_excluded_runtime_fields(backtest_spy):
    class Parameters(ModelParams):
        count: int = Field(exclude=True, gt=0)

    class Analysis(Parameters, ModelAnalysisParams):
        pass

    class Model(ModelAlgo[Parameters, State]):
        pass

    params = Analysis(**DATES, count=9)
    assert "count" not in params.model_dump()
    result = params.run(Model)
    assert backtest_spy[0][0]["algos"][0].params.count == 9
    assert result.parameters.count == 9


def test_stage_run_preserves_nested_runtime_subtypes(backtest_spy):
    class Option(BaseModel):
        common: int = 1

    class AdvancedOption(Option):
        threshold: int

    class Parameters(ModelParams):
        option: Option

    class Analysis(Parameters, ModelAnalysisParams):
        pass

    class Model(ModelAlgo[Parameters, State]):
        pass

    option = AdvancedOption(threshold=99)
    params = Analysis(**DATES, option=option)
    result = params.run(Model)
    actual = backtest_spy[0][0]["algos"][0].params.option
    assert actual is option
    assert type(actual) is AdvancedOption
    assert actual.threshold == result.parameters.option.threshold == 99
    assert result.parameters.option is not option


def test_stage_run_preserves_canonical_runtime_aliases(backtest_spy):
    class Parameters(ModelParams):
        count: int = Field(alias="n_select")

    class Analysis(Parameters, ModelAnalysisParams):
        pass

    class Model(ModelAlgo[Parameters, State]):
        pass

    params = Analysis(**DATES, n_select=17)
    result = params.run(Model)
    assert backtest_spy[0][0]["algos"][0].params.count == 17
    assert result.parameters.count == 17


def test_real_backtest_api_with_offline_engine_retains_parameter_subtype(settings):
    from scheme.execute.strategy.api import run_backtest

    class Model(ModelAlgo[ModelParams, State]):
        pass

    class Analysis(StrategyAnalysisParams):
        count: int

    params = Analysis(**DATES, count=3, batch_days=2, config={"cash": 100, "nested": [1]})
    ctx = State()
    algos = [Model(ModelParams())]
    calls = []

    class Engine:
        def __init__(self, selected_settings, parameters, context, selected_algos):
            assert (selected_settings, parameters, context, selected_algos) == (
                settings, params, ctx, algos,
            )

        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *error):
            calls.append("exit")

        def run(self, batch):
            calls.append(batch)

        def get_daily_total_portfolios(self):
            return []

        def get_trade_details(self):
            return []

        def get_daily_position(self):
            return []

        def get_daily_trading_statistics(self):
            return []

    result = run_backtest(algos, ctx, params, settings=settings, engine_type=Engine)
    assert type(result) is BacktestResult
    assert type(result.parameters) is Analysis
    assert result.parameters == params
    assert result.parameters is not params
    assert all(getattr(result, name).empty for name in result.filenames)
    assert calls == ["enter", np.timedelta64(2, "D"), "exit"]
    params.config["nested"].append(2)
    assert result.parameters.config["nested"] == [1]


@pytest.mark.parametrize("quotes", [
    [],
    [("2025-01-02", 100.0), ("2025-01-04", 110.0)],
    [("2025-01-01", 0.0), ("2025-01-02", 100.0), ("2025-01-03", np.inf),
     ("2025-01-04", 110.0)],
    [("2025-01-02", np.nan)],
])
def test_backtest_preserves_results_with_empty_or_sparse_benchmark(monkeypatch, settings, quotes):
    from scheme.execute.strategy import api

    parameters = StrategyAnalysisParams(
        start="2025-01-01", end="2025-01-06", benchmark="000300.SH",
    )
    benchmark = pd.DataFrame(quotes, columns=["time", "close"])
    benchmark["time"] = pd.to_datetime(benchmark.time)
    request = Mock(return_value=nullcontext(SimpleNamespace(data=benchmark)))
    warning = Mock()
    monkeypatch.setattr(api, "query", request)
    monkeypatch.setattr(api.log, "warning", warning)
    engine = Mock()
    engine.__enter__ = Mock(return_value=engine)
    engine.__exit__ = Mock(return_value=False)
    engine.get_daily_total_portfolios.return_value = [
        Mock(model_dump=Mock(return_value={"tradeDate": day.date(), "netValue": 1.0}))
        for day in pd.date_range("2025-01-01", periods=5)
    ]
    for name in ("get_trade_details", "get_daily_position", "get_daily_trading_statistics"):
        getattr(engine, name).return_value = []
    result = api._run([], State(), parameters, settings, Mock(return_value=engine))
    portfolios = result.daily_portfolios
    assert len(portfolios) == 5
    assert portfolios.netValue.tolist() == [1.0] * 5
    assert request.call_args.args[0]["end_date"] == "2025-01-05"
    if not quotes:
        assert "benchmarkClosePrice" not in portfolios
        assert "benchmarkNetValue" not in portfolios
        warning.assert_called_once()
    elif benchmark.close.gt(0).any():
        assert portfolios.benchmarkClosePrice.iloc[[0, 2, 4]].isna().all()
        assert portfolios.benchmarkNetValue.iloc[[0, 2, 4]].isna().all()
        assert portfolios.benchmarkNetValue.iloc[[1, 3]].tolist() == pytest.approx([1.0, 1.1])
    else:
        assert portfolios.benchmarkClosePrice.isna().all()
        assert portfolios.benchmarkNetValue.isna().all()


def test_strategy_validates_context_only_at_backtest_entry(monkeypatch, settings):
    from scheme.execute.strategy import api, assembly

    class Model(ModelAlgo[ModelParams, State]):
        pass

    validate = Mock(wraps=assembly.validate_context)
    monkeypatch.setattr(assembly, "validate_context", validate)
    monkeypatch.setattr(api, "validate_context", validate)
    strategy = Strategy(State(), params=StrategyParams(**DATES), model=Model)
    validate.assert_not_called()
    parameters = StrategyAnalysisParams(**DATES)
    result = empty_backtest(parameters)
    run = Mock(return_value=result)
    monkeypatch.setattr(api, "_run", run)

    assert api.run_backtest(strategy.algos, strategy.ctx, parameters, settings=settings) is result
    assert [call.args for call in validate.call_args_list] == [
        (algo, strategy.ctx) for algo in strategy.algos
    ]
    run.assert_called_once()


def test_run_backtest_validates_context_once_before_creating_engine(monkeypatch, settings):
    from scheme.execute.strategy import api

    class Model(ModelAlgo[ModelParams, State]):
        pass

    engine_type = Mock(side_effect=AssertionError("invalid context must not start the engine"))
    with pytest.raises(TypeError, match="State"):
        api.run_backtest([Model(ModelParams())], OtherState(), StrategyAnalysisParams(**DATES),
                         settings=settings, engine_type=engine_type)
    engine_type.assert_not_called()


@pytest.fixture
def frozen_factor_task(tmp_path, monkeypatch):
    from scheme.execute.factor import task

    class Parameters(FactorParams):
        window: int = Field(gt=0)

    class FrozenFactor(Factor[Parameters]):
        def compute(self, start: date, end: date) -> pd.DataFrame:
            raise AssertionError("frozen task test must not compute or query")

    data = dict(
        kind="factor", environment={"lockfile": str(tmp_path / "uv.lock")},
        output=str(tmp_path / "unused-output"),
        factor=dict(package="factor_fixture", version="1.0.0", wheel="fixture.whl",
                    entry="factor_fixture:Factor", params={"window": "17"}),
        analysis=FactorAnalysisParams(**DATES, columns=["f"], groups=3).model_dump(mode="json"),
    )
    versions = {"scheme": "1.0.1"}
    report = empty_factor_report(FactorAnalysisParams.model_validate(data["analysis"]))
    spies = {
        "check_output": Mock(),
        "installed_versions": Mock(return_value=versions),
        "load_component": Mock(return_value=FrozenFactor),
        "analyze_factors": Mock(return_value=report),
        "save_run": Mock(),
    }
    for name, spy in spies.items():
        monkeypatch.setattr(task, name, spy)
    return task, data, Parameters, FrozenFactor, spies, versions, report


@pytest.mark.parametrize("conflicting_universe", [False, True])
def test_frozen_factor_task_keeps_custom_runtime_without_analysis_leak(
    frozen_factor_task, conflicting_universe,
):
    task, data, model, factor_type, spies, versions, report = frozen_factor_task
    if conflicting_universe:
        data["factor"]["params"]["universe"] = {"pool": StockPool.CSI300}
    original = deepcopy(data)
    assert task.run(data) == 0
    factor, analysis = spies["analyze_factors"].call_args.args
    assert type(factor) is factor_type
    assert type(factor.params) is model
    assert factor.params.window == 17
    assert set(factor.params.model_dump()) == {"universe", "window"}
    assert not hasattr(factor.params, "start")
    assert not hasattr(factor.params, "end")
    assert (analysis.start, analysis.end) == (date(2025, 1, 1), date(2025, 1, 3))
    assert factor.params.universe == analysis.universe
    assert type(analysis) is FactorAnalysisParams
    assert not hasattr(analysis, "window")
    saved, saved_report, saved_versions = spies["save_run"].call_args.args
    assert saved_report is report
    assert saved_versions is versions
    assert saved.factor.params == factor.params.model_dump(mode="json")
    assert FactorTask.model_validate_json(saved.model_dump_json()) == saved
    assert spies["save_run"].call_args.kwargs == {}
    assert spies["load_component"].call_args.args == (saved.factor, Factor)
    spies["installed_versions"].assert_called_once_with()
    spies["check_output"].assert_called_once_with(saved)
    assert data == original
    assert not saved.output.exists()


@pytest.mark.parametrize("field", ["window", "days"])
def test_frozen_factor_task_accepts_canonical_and_alias_runtime_fields(
    frozen_factor_task, field,
):
    task, data, _, _, spies, _, _ = frozen_factor_task

    class Parameters(FactorParams):
        window: int = Field(alias="days", gt=0)

    class AliasedFactor(Factor[Parameters]):
        def compute(self, start: date, end: date) -> pd.DataFrame:
            raise AssertionError("frozen task test must not compute or query")

    spies["load_component"].return_value = AliasedFactor
    data["factor"]["params"] = {field: 17}
    assert task.run(data) == 0
    assert spies["analyze_factors"].call_args.args[0].params.window == 17
    assert spies["save_run"].call_args.args[0].factor.params["window"] == 17


@pytest.mark.parametrize("alias", [
    "pool_settings", AliasChoices("pool_settings", "selection"), AliasPath("shared", "universe"),
])
def test_frozen_factor_analysis_overrides_only_shared_universe_aliases(frozen_factor_task, alias):
    task, data, _, _, spies, _, _ = frozen_factor_task

    class Parameters(FactorParams):
        universe: Universe = Field(validation_alias=alias)

    class AliasedFactor(Factor[Parameters]):
        def compute(self, start: date, end: date) -> pd.DataFrame:
            raise AssertionError("frozen task test must not compute or query")

    spies["load_component"].return_value = AliasedFactor
    data["factor"]["params"] = {
        "pool_settings": {"pool": StockPool.CSI300}, "selection": {"pool": StockPool.CSI500},
        "shared": {"universe": {"pool": StockPool.SSE50}},
    }
    original = deepcopy(data)
    assert task.run(data) == 0
    factor, analysis = spies["analyze_factors"].call_args.args
    assert factor.params.universe == analysis.universe
    assert set(factor.params.model_dump()) == {"universe"}
    assert (analysis.start, analysis.end) == (date(2025, 1, 1), date(2025, 1, 3))
    assert data == original


@pytest.mark.parametrize("runtime", [{}, {"window": 0}, {"window": "invalid"}])
def test_frozen_factor_task_validates_required_custom_runtime(frozen_factor_task, runtime):
    task, data, _, _, spies, _, _ = frozen_factor_task
    data["factor"]["params"] = runtime
    with pytest.raises(ValidationError, match="window"):
        task.run(data)
    spies["analyze_factors"].assert_not_called()
    spies["save_run"].assert_not_called()


def test_risk_parity_equal_risk():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20000, 2)) * [1.0, 3.0]
    weights, reason = risk_parity(pd.DataFrame(x))
    assert reason is None
    assert weights == pytest.approx([0.75, 0.25], abs=0.015)
    covariance = np.cov(x.T)
    risk = weights * (covariance @ weights)
    assert risk[0] / risk[1] == pytest.approx(1.0, abs=0.01)


def test_risk_parity_solves_short_singular_history():
    returns = pd.DataFrame({"first": [-0.01, 0.01], "second": [-0.03, 0.03]})
    weights, reason = risk_parity(returns)
    assert reason is None
    assert weights == pytest.approx([0.75, 0.25], abs=0.001)


@pytest.mark.parametrize("rows", [[], [[0.01, 0.02]], [[0.01, None], [None, 0.02]]])
def test_risk_parity_needs_two_complete_return_rows(rows):
    weights, reason = risk_parity(pd.DataFrame(rows, columns=["a", "b"], dtype=float))
    assert weights == pytest.approx([0.5, 0.5])
    assert reason == "历史样本不足"


def test_risk_parity_falls_back_only_when_solver_fails(monkeypatch):
    solver = Mock(return_value=SimpleNamespace(success=False))
    monkeypatch.setattr("scheme.base.projects.optimize.default.minimize", solver)
    weights, reason = risk_parity(pd.DataFrame({"a": [0.01, -0.01], "b": [0.02, -0.02]}))
    assert weights == pytest.approx([0.5, 0.5])
    assert reason == "风险平价求解未收敛"
    solver.assert_called_once()


@pytest.mark.parametrize("scalar", [np.int64(1), np.float32(-1.0)])
def test_risk_parity_accepts_numpy_real_signals(monkeypatch, scalar):
    from scheme.base.projects.optimize import default

    allocator = default.RiskParity[State]()
    allocator.backtest = SimpleNamespace(time=pd.Timestamp("2025-01-02"))
    history = pd.DataFrame({
        "time": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-01"]),
        "code": ["000001.SZ"] * 3, "close": [10.0, 11.0, 10.5], "adj_factor": [1.0] * 3,
    })
    monkeypatch.setattr(default, "query", lambda _: nullcontext(SimpleNamespace(data=history)))
    target = allocator.allocate({"000001.XSHE": scalar})
    assert target == {"000001.XSHE": pytest.approx(float(np.sign(scalar)) * 0.98)}


@pytest.mark.parametrize("in_place", [False, True])
def test_risk_parity_allocate_normalizes_symbol_aliases(monkeypatch, in_place):
    from scheme.base.projects.optimize import default

    allocator = default.RiskParity[State]()
    allocator.ctx = State()
    allocator.backtest = SimpleNamespace(time=pd.Timestamp("2025-01-02"))
    history = pd.DataFrame({
        "time": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-01"]).repeat(2),
        "code": ["000001.SZ", "600000.SH"] * 3,
        "close": [100.0, 100.0, 99.0, 97.0, 99.99, 99.91], "adj_factor": [1.0] * 6,
    })
    query = Mock(side_effect=lambda _: nullcontext(SimpleNamespace(data=history.copy())))
    monkeypatch.setattr(default, "query", query)
    expected = allocator.allocate({"000001.XSHE": 1.0, "600000.XSHG": 1.0})
    assert expected == pytest.approx({"000001.XSHE": 0.735, "600000.XSHG": 0.245}, abs=0.001)
    signals = {"000001.SZ": 1.0, "600000.SH": 1.0}
    if in_place:
        allocator.ctx.signal = {}
        allocator.ctx.signal.update(signals)
        assert allocator.process()
        actual = allocator.ctx.target
        assert allocator.ctx.signal is None
    else:
        actual = allocator.allocate(signals)
    assert actual == expected
    assert allocator.diagnostics[-1]["method"] == "risk_parity"
    assert query.call_args.args[0]["codes"] == ["000001.SZ", "600000.SH"]
    assert signals == {"000001.SZ": 1.0, "600000.SH": 1.0}


@pytest.mark.parametrize("in_place", [False, True])
def test_risk_parity_allocate_rejects_duplicate_symbol_aliases(monkeypatch, in_place):
    from scheme.base.projects.optimize import default

    allocator = default.RiskParity[State]()
    allocator.ctx = State()
    query = Mock(side_effect=AssertionError("duplicate symbols must not query history"))
    monkeypatch.setattr(default, "query", query)
    signals = {"000001.SZ": 1.0, "000001.XSHE": 0.0}
    if in_place:
        allocator.ctx.signal = {}
        allocator.ctx.signal.update(signals)
    with pytest.raises(ValueError, match="证券代码规范化后重复"):
        if in_place:
            allocator.process()
        else:
            allocator.allocate(signals)
    query.assert_not_called()


def test_default_risk_parity_allows_zero_gross_exposure():
    from scheme.base.projects.optimize.default import OptimizeParams

    assert OptimizeParams(gross_exposure=0).gross_exposure == 0


@pytest.fixture
def default_control():
    from scheme.base.projects.control.default import NoControl

    control = NoControl[State]()
    control.ctx = State()
    control.backtest = SimpleNamespace(
        time=pd.Timestamp("2025-01-02"), get_prices=Mock(return_value={}),
        get_position=Mock(return_value={}), session=SimpleNamespace(run=Mock(return_value=100_000)),
    )
    return control


@pytest.mark.parametrize("in_place", [False, True])
def test_target_orders_normalizes_symbol_aliases(default_control, monkeypatch, in_place):
    from scheme.base.projects.control import algo

    warning = Mock()
    monkeypatch.setattr(algo.log, "warning", warning)
    default_control.backtest.get_prices.return_value = {"000001.XSHE": 10.0, "600000.XSHG": 20.0}
    expected = default_control.target_orders({"000001.XSHE": 0.5, "600000.XSHG": -0.3})
    target = {"000001.SZ": 0.5, "600000.SH": -0.3}
    if in_place:
        default_control.ctx.target = {}
        default_control.ctx.target.update(target)
        assert default_control.process()
        actual = default_control.ctx.orders
        assert default_control.ctx.target is None
    else:
        actual = default_control.target_orders(target)
    assert actual == expected
    assert [(order.symbol, order.quantity) for order in actual] == [
        ("000001.XSHE", 5000), ("600000.XSHG", 1500),
    ]
    default_control.backtest.get_prices.assert_called_with(["000001.XSHE", "600000.XSHG"])
    warning.assert_not_called()
    assert target == {"000001.SZ": 0.5, "600000.SH": -0.3}


@pytest.mark.parametrize("in_place", [False, True])
def test_target_orders_rejects_duplicate_symbol_aliases(default_control, in_place):
    target = {"000001.SZ": 0.5, "000001.XSHE": 0.0}
    if in_place:
        default_control.ctx.target = {}
        default_control.ctx.target.update(target)
    with pytest.raises(ValueError, match="证券代码规范化后重复"):
        if in_place:
            default_control.process()
        else:
            default_control.target_orders(target)
    default_control.backtest.get_prices.assert_not_called()
    default_control.backtest.get_position.assert_not_called()
    default_control.backtest.session.run.assert_not_called()


@pytest.mark.parametrize("price", [None, np.nan, np.inf, -np.inf, 0.0, -10.0])
@pytest.mark.parametrize("position", [None, (300, 0), (0, 200), (300, 200)])
def test_target_orders_skips_missing_prices_without_changing_positions(
    default_control, monkeypatch, price, position,
):
    from backtest import Direction

    from scheme.base.projects.control import algo

    warning = Mock()
    monkeypatch.setattr(algo.log, "warning", warning)
    prices = {"600000.XSHG": 20.0}
    if price is not None:
        prices["000001.XSHE"] = price
    positions = {"600000.XSHG": SimpleNamespace(longPosition=500, shortPosition=0)}
    if position is not None:
        positions["000001.XSHE"] = SimpleNamespace(
            longPosition=position[0], shortPosition=position[1],
        )
    default_control.backtest.get_prices.return_value = prices
    default_control.backtest.get_position.return_value = positions
    orders = default_control.target_orders({"000001.XSHE": 0.5, "600000.XSHG": -0.3})
    assert [(order.symbol, order.direction, order.quantity) for order in orders] == [
        ("600000.XSHG", Direction.SELL_CLOSE, 500), ("600000.XSHG", Direction.SELL_OPEN, 1500),
    ]
    warning.assert_called_once()
    assert warning.call_args.args[1] == 1
    default_control.backtest.session.run.assert_called_once()
    if position is not None:
        assert vars(positions["000001.XSHE"]) == {
            "longPosition": position[0], "shortPosition": position[1],
        }


def test_target_orders_without_any_valid_prices_only_warns_once(default_control, monkeypatch):
    from scheme.base.projects.control import algo

    warning = Mock()
    monkeypatch.setattr(algo.log, "warning", warning)
    default_control.backtest.get_position.return_value = {
        "000001.XSHE": SimpleNamespace(longPosition=500, shortPosition=100),
    }
    assert default_control.target_orders({"000001.XSHE": 0.5, "600000.XSHG": -0.3}) == []
    warning.assert_called_once()
    assert warning.call_args.args[1] == 2
    default_control.backtest.session.run.assert_not_called()


def test_target_orders_still_closes_zero_and_omitted_targets(default_control):
    from backtest import Direction

    default_control.backtest.get_position.return_value = {
        "000001.XSHE": SimpleNamespace(longPosition=300, shortPosition=0),
        "600000.XSHG": SimpleNamespace(longPosition=0, shortPosition=200),
    }
    orders = default_control.target_orders({"000001.XSHE": 0.0})
    assert [(order.symbol, order.direction, order.quantity) for order in orders] == [
        ("000001.XSHE", Direction.SELL_CLOSE, 300), ("600000.XSHG", Direction.BUY_CLOSE, 200),
    ]
    default_control.backtest.get_prices.assert_not_called()
    default_control.backtest.session.run.assert_not_called()


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf])
def test_target_orders_still_rejects_nonfinite_weights(default_control, weight):
    with pytest.raises(ValueError, match="目标权重必须是有限数值"):
        default_control.target_orders({"000001.SZ": weight})
    default_control.backtest.get_prices.assert_not_called()


@pytest.mark.parametrize("equity", [0.0, -1.0, np.nan, np.inf])
def test_target_orders_still_rejects_invalid_equity(default_control, equity):
    default_control.backtest.get_prices.return_value = {"000001.XSHE": 10.0}
    default_control.backtest.session.run.return_value = equity
    with pytest.raises(ValueError, match="账户权益必须为正数"):
        default_control.target_orders({"000001.XSHE": 0.5})


@pytest.mark.parametrize("base,field,method", [
    (OptimizeAlgo, "signal", "on_signal"),
    (ControlAlgo, "target", "on_target"),
    (ExecutionAlgo, "orders", "on_orders"),
])
def test_stages_consume_already_normalized_context_message_once(base, field, method):
    from backtest import Direction, MarketOrder

    received = []
    Consumer = type("Consumer", (base,), {
        method: lambda self, value: received.append(value),
    })
    context = State()
    value = [MarketOrder(
        symbol="000001.SZ", time=pd.Timestamp("2025-01-02").to_pydatetime(),
        quantity=100, direction=Direction.BUY_OPEN,
    )] if field == "orders" else {"000001.SZ": 1.0}
    setattr(context, field, value)
    message = getattr(context, field)
    assert (message[0].symbol if field == "orders" else next(iter(message))) == "000001.XSHE"
    consumer = Consumer(ModelParams())
    consumer.ctx = context
    assert consumer.process()
    assert len(received) == 1 and received[0] is message
    assert getattr(context, field) is None
    assert not consumer.process()


@pytest.mark.parametrize(
    "updates",
    [
        {"end": "2025-01-01"},
        {"batch_days": 0},
        {"config": {"startDate": "2025-01-01"}},
        {"config": {"strategyGroup": "future"}},
        {"symbols": ["000001.SZ", "000001.XSHE"]},
    ],
)
def test_backtest_rejects_invalid_config(updates):
    with pytest.raises(ValidationError):
        StrategyAnalysisParams.model_validate({**DATES, "symbols": ["000001.SZ"], **updates})


def test_entry_must_belong_to_distribution():
    with pytest.raises(ValueError, match="不属于"):
        load_entry(distribution("scheme"), "pathlib:Path")


def test_cli_failure_returns_nonzero(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"kind":"factor"}')
    assert main(["run", "--input", str(source), "--output", str(tmp_path / "report")]) == 1


def test_unknown_input_field_rejected():
    with pytest.raises(ValidationError):
        StrategyTask.model_validate({"kind": "strategy", "typo": True})
