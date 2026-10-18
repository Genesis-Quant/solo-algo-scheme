import json
import re
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from runtime.apps.query import FactorQuery

from scheme.base import FactorAnalysisParams, FactorParams, FactorReportForm
from scheme.config import DolphinSettings
from scheme.execute.factor.api import _return_derivatives
from scheme.execute.schema import FactorTask

DATES = {"start": "2025-01-03", "end": "2025-01-09"}
PRICE_COLUMNS = ["open", "close", "open_hfq", "close_hfq"]


def evaluate_returns(prices, *, period=1, max_lag=1, price_column="open_hfq", calendar=None):
    # Runtime supplies a code x trading-day grid, including NULL rows for suspended stocks.
    if calendar is None:
        calendar = pd.DatetimeIndex(prices.time.unique()).sort_values()
    grid = pd.MultiIndex.from_product(
        [sorted(prices.code.unique()), calendar], names=["code", "time"],
    )
    frame = prices.set_index(["code", "time"]).reindex(grid).reset_index()

    def evaluate(value):
        if isinstance(value, str):
            return frame[value]
        if not isinstance(value, dict):
            return value
        fields = value["fields"]
        if value["op"] == "unary.shift":
            return frame.groupby("code", sort=False)[fields["col"]].shift(value["params"]["periods"])
        left, right = evaluate(fields["left"]), evaluate(fields["right"])
        if value["op"] == "binary.div":
            return left / right
        assert value["op"] == "binary.sub"
        return left - right

    for name, node in _return_derivatives(period, max_lag, price_column).items():
        frame[name] = evaluate(node)
    return frame


@pytest.mark.parametrize("period,max_lag,expected", [
    (1, 1, ["P1L1"]),
    (1, 3, ["P1L1", "P1L2", "P1L3"]),
    (5, 3, ["P5L1", "P5L2", "P5L3"]),
])
@pytest.mark.parametrize("price_column", PRICE_COLUMNS)
def test_parameters_generate_ordered_holding_returns_and_metadata(
    period, max_lag, expected, price_column,
):
    params = FactorAnalysisParams(
        **DATES, columns=["score"], period=period, max_lag=max_lag, price_column=price_column,
    )
    assert params.return_columns == expected
    assert list(params.return_specs) == expected
    for lag, name in enumerate(expected, start=1):
        assert params.return_specs[name] == {
            "kind": "simple", "periods": period, "lag": lag,
            "price_column": price_column, "turnover_period": period,
        }
    assert "return_columns" not in params.model_dump()
    assert "return_specs" not in params.model_dump()
    assert FactorAnalysisParams.model_validate_json(params.model_dump_json()) == params
    form = FactorReportForm(**params.model_dump(exclude={"universe"}))
    assert form.build().model_dump(exclude={"universe"}) == params.model_dump(exclude={"universe"})


@pytest.mark.parametrize("model", [FactorAnalysisParams, FactorReportForm])
def test_analysis_and_form_share_price_and_positive_integer_defaults(model):
    params = model(**DATES, columns=["score"])
    assert (params.period, params.price_column, params.max_lag) == (1, "open_hfq", 1)
    properties = model.model_json_schema()["properties"]
    for name in ("period", "max_lag"):
        schema = properties[name]
        assert schema["type"] == "integer" and schema["minimum"] == 1 and schema["default"] == 1
    assert properties["price_column"]["enum"] == PRICE_COLUMNS
    assert properties["price_column"]["default"] == "open_hfq"


@pytest.mark.parametrize("field", ["period", "max_lag"])
@pytest.mark.parametrize("value", [
    0, -1, True, False, 1.0, 1.5, "2", None, [1, 2], np.nan, np.inf,
])
@pytest.mark.parametrize("model", [FactorAnalysisParams, FactorReportForm])
def test_holding_period_and_max_lag_require_strict_positive_integers(model, field, value):
    with pytest.raises(ValidationError):
        model(**DATES, columns=["score"], **{field: value})


@pytest.mark.parametrize("value", ["high", "open_qfq", "adjusted_open", "OPEN", "", 1, None])
@pytest.mark.parametrize("model", [FactorAnalysisParams, FactorReportForm])
def test_price_column_accepts_only_runtime_price_literals(model, value):
    with pytest.raises(ValidationError, match="price_column"):
        model(**DATES, columns=["score"], price_column=value)


@pytest.mark.parametrize("field,value", [("period", 5), ("price_column", "close"), ("max_lag", 3)])
def test_factor_own_params_do_not_gain_analysis_fields(field, value):
    assert set(FactorParams.model_fields) == {"universe"}
    with pytest.raises(ValidationError, match=field):
        FactorParams(**{field: value})


@pytest.mark.parametrize("model", [FactorAnalysisParams, FactorReportForm])
def test_old_holding_periods_are_not_reinterpreted_as_lags(model):
    with pytest.raises(ValidationError, match="return_periods"):
        model(**DATES, columns=["score"], return_periods=[1, 5, 20])


@pytest.mark.parametrize("column", ["P5L1", "P5L2", "P5L3", "time", "code", "date", "symbol", "circ_mv"])
def test_factor_columns_cannot_shadow_generated_or_join_columns(column):
    with pytest.raises(ValidationError, match="重名"):
        FactorAnalysisParams(**DATES, columns=[column], period=5, max_lag=3)
    form = FactorReportForm(**DATES, columns=[column], period=5, max_lag=3)
    with pytest.raises(ValidationError, match="重名"):
        form.build()


@pytest.mark.parametrize("column", ["adjusted_open", "adjusted_close", "P1L1", "P5L4"])
def test_factor_columns_may_use_names_not_generated_by_this_analysis(column):
    params = FactorAnalysisParams(**DATES, columns=[column], period=5, max_lag=3)
    assert params.columns == [column]
    assert FactorReportForm(**DATES, columns=[column], period=5, max_lag=3).build().columns == [column]


@pytest.mark.parametrize("price_column", PRICE_COLUMNS)
def test_p5l1_to_p5l3_use_five_day_holding_returns_after_each_lag(price_column):
    days = pd.bdate_range("2025-01-03", periods=10)
    prices = pd.DataFrame({
        "time": days, "code": ["000001.SZ"] * len(days),
        "open": [90.0, 55.0, 65.0, 70.0, 80.0, 85.0, 95.0, 100.0, 110.0, 120.0],
        "close": [100.0, 60.0, 66.0, 77.0, 83.0, 91.0, 99.0, 107.0, 115.0, 127.0],
        # Real Runtime hfq fields deliberately differ from raw price * adj_factor.
        "open_hfq": [180.0, 111.0, 132.0, 143.0, 167.0, 179.0, 197.0, 211.0, 233.0, 257.0],
        "close_hfq": [201.0, 123.0, 135.0, 159.0, 173.0, 191.0, 209.0, 227.0, 251.0, 281.0],
        "adj_factor": [7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0],
    })
    frame = evaluate_returns(
        prices.sample(frac=1, random_state=5), period=5, max_lag=3, price_column=price_column,
    )
    values = prices[price_column].tolist()
    columns = ["P5L1", "P5L2", "P5L3"]
    assert frame.loc[0, columns].tolist() == pytest.approx([
        values[6] / values[1] - 1, values[7] / values[2] - 1, values[8] / values[3] - 1,
    ])
    assert frame.loc[1, columns].tolist() == pytest.approx([
        values[7] / values[2] - 1, values[8] / values[3] - 1, values[9] / values[4] - 1,
    ])
    assert frame.loc[0, "P5L1"] != pytest.approx(values[1] / values[0] - 1)
    assert pd.notna(frame.loc[1, "P5L3"]) and pd.isna(frame.loc[2, "P5L3"])
    assert frame.loc[4:, columns].isna().all().all()


@pytest.mark.parametrize("price_column", PRICE_COLUMNS)
def test_returns_never_depend_on_signal_day_prices_or_other_price_fields(price_column):
    prices = pd.DataFrame({
        "time": pd.bdate_range("2025-01-03", periods=4), "code": ["A"] * 4,
        "open": [9.0, 11.0, 13.0, 17.0], "close": [10.0, 12.0, 15.0, 19.0],
        "open_hfq": [21.0, 23.0, 29.0, 31.0], "close_hfq": [25.0, 27.0, 33.0, 37.0],
        "adj_factor": [1.0, 2.0, 3.0, 4.0],
    })
    original = evaluate_returns(prices, max_lag=2, price_column=price_column)
    changed = prices.copy()
    changed.loc[0, PRICE_COLUMNS] = np.nan
    changed.loc[:, [name for name in PRICE_COLUMNS if name != price_column]] = np.nan
    changed["adj_factor"] = np.nan
    actual = evaluate_returns(changed, max_lag=2, price_column=price_column)
    assert actual.loc[0, ["P1L1", "P1L2"]].tolist() == pytest.approx(
        original.loc[0, ["P1L1", "P1L2"]].tolist(),
    )
    assert not any(name.startswith("r") for name in actual.columns)


def test_missing_quotes_use_trading_grid_not_calendar_days_or_valid_quote_positions():
    # Jan 1 is a market holiday; the weekend and holiday must not occupy grid positions.
    days = pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    prices = pd.DataFrame({
        "time": [*days, *days], "code": ["A"] * 5 + ["B"] * 5,
        "open_hfq": [9.0, np.nan, 13.0, 17.0, 19.0, 20.0, 22.0, 26.0, 34.0, 38.0],
    })
    # A has a NULL quote and B has no row on the second trading day.
    prices = prices[~((prices.code == "B") & (prices.time == days[1]))]
    frame = evaluate_returns(prices, max_lag=3, calendar=days).set_index(["code", "time"])
    for code in ("A", "B"):
        first = frame.loc[(code, days[0])]
        assert pd.isna(first.P1L1)
        assert first.P1L2 == pytest.approx(17 / 13 - 1)
        assert first.P1L3 == pytest.approx(19 / 17 - 1)
        assert frame.loc[(code, days[3]), ["P1L1", "P1L2", "P1L3"]].isna().all()
        assert frame.loc[(code, days[4]), ["P1L1", "P1L2", "P1L3"]].isna().all()
    assert len(frame) == 2 * len(days)
    assert frame.loc[("B", days[1]), "P1L1"] == pytest.approx(34 / 26 - 1)
    # Joining a sparse factor panel only after computing returns must not shrink this axis.
    sparse = frame[frame.index.get_level_values("time").isin(days[[0, 2, 4]])]
    assert sparse.loc[("A", days[0]), "P1L2"] == pytest.approx(17 / 13 - 1)


@pytest.mark.parametrize("price_column", PRICE_COLUMNS)
def test_generated_graph_reads_only_selected_runtime_field_at_strictly_future_offsets(price_column):
    derivatives = _return_derivatives(5, 3, price_column)
    query = FactorQuery.model_validate({
        "start_date": DATES["start"], "end_date": DATES["end"], "codes": ["000001.SZ"],
        "factors": [], "derivatives": derivatives,
    })
    assert set(query.source_factors()) == {price_column}
    assert list(derivatives) == ["P5L1", "P5L2", "P5L3"]
    for lag, node in enumerate(derivatives.values(), start=1):
        assert node["op"] == "binary.sub" and node["fields"]["right"] == 1.0
        division = node["fields"]["left"]
        assert division["op"] == "binary.div"
        for operand, offset in (("left", lag + 5), ("right", lag)):
            assert division["fields"][operand] == {
                "type": "TS", "op": "unary.shift",
                "fields": {"col": price_column}, "params": {"periods": -offset},
            }
    assert "adj_factor" not in json.dumps(derivatives)
    assert "r0o" not in derivatives


def test_turnover_uses_single_holding_period_on_full_calendar_not_sparse_factor_dates(monkeypatch):
    from scheme.execute.factor import api

    days = pd.bdate_range("2025-01-03", periods=10)
    params = FactorAnalysisParams(
        start=days[0].date(), end=days[-1].date(), columns=["score"], period=5, max_lag=3,
    )
    panel = pd.DataFrame(
        {"score": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_product(
            [days[[0, 2, 5, 7]], ["000001.XSHE"]], names=["date", "symbol"],
        ),
    )
    prices = pd.DataFrame({
        "time": days, "code": ["000300.SH"] * len(days), "open_hfq": np.arange(10.0, 20.0),
    })
    factor = Mock()
    factor.compute.return_value = panel
    session = SimpleNamespace(run=Mock(return_value=pd.DataFrame()), upload=Mock())
    request = Mock(return_value=nullcontext(SimpleNamespace(data_ref="offlineReturns")))
    monkeypatch.setattr(api, "create_session", lambda _: nullcontext(session))
    monkeypatch.setattr(api, "query", request)
    api.analyze_factors(
        factor, params,
        settings=DolphinSettings(host="unused.invalid", username="offline", password="unused"),
    )

    uploaded = session.upload.call_args.args[0]
    assert uploaded["soloPeriods"].tolist() == [5]
    assert uploaded["soloPeriods"].dtype == np.dtype("int32")
    assert uploaded["soloReturns"].tolist() == ["P5L1", "P5L2", "P5L3"]
    assert uploaded["soloCalendarSymbol"] == "000300.SH"
    scripts = [call.args[0] for call in session.run.call_args_list]
    setup = scripts[0]
    assert "soloCalendar = exec time from offlineReturns where code = soloCalendarSymbol" in setup
    assert "soloPrices = select * from offlineReturns where time in soloCalendar" in setup
    assert "dropColumns!" not in setup
    session.run.assert_any_call(
        'soloFactorGroupTurnover(soloFactorProcessed, soloFactors, soloPeriods, '
        'soloGroups, soloSelect, "time", "code", soloCalendar)',
    )
    assert all("distinct(soloFactorSource.time)" not in script for script in scripts)
    # The actual 5-day predecessor of day 5 is day 0. A compact factor-observation axis
    # has only four entries and could not supply this pair to Runtime's turnover helper.
    calendar = pd.DatetimeIndex(prices.loc[prices.code == uploaded["soloCalendarSymbol"], "time"])
    observations = pd.DatetimeIndex(uploaded["soloFactorSource"].time.unique())
    assert len(calendar) == 10 and len(observations) == 4
    assert calendar[calendar.get_loc(days[5]) - uploaded["soloPeriods"][0]] == days[0]
    assert days[1] in calendar and days[1] not in observations
    assert request.call_args.args[0]["end_date"] == days[-1].date().isoformat()


def test_factor_task_serializes_analysis_fields_but_not_derived_metadata(tmp_path):
    params = FactorAnalysisParams(
        **DATES, columns=["score"], period=5, price_column="close_hfq", max_lag=3,
    )
    task = FactorTask.model_validate({
        "environment": {"lockfile": str(tmp_path / "uv.lock")}, "output": str(tmp_path / "output"),
        "factor": {"package": "factor-test", "version": "1.0.0", "wheel": "test.whl",
                   "entry": "factor_test:Factor"},
        "analysis": params.model_dump(mode="json"),
    })
    assert task.analysis.return_columns == ["P5L1", "P5L2", "P5L3"]
    serialized = task.model_dump()["analysis"]
    assert (serialized["period"], serialized["price_column"], serialized["max_lag"]) == (5, "close_hfq", 3)
    assert "return_specs" not in serialized
    assert "return_periods" not in serialized


def test_standalone_html_embeds_holding_return_specs_without_ipython(monkeypatch):
    module = import_module("scheme.report.html")
    monkeypatch.setattr(module, "_assets", lambda: {})
    report = SimpleNamespace(
        report_kind="factor", filenames={},
        parameters=FactorAnalysisParams(**DATES, columns=["score"], period=5, max_lag=3),
    )
    document = module.report_html(report)
    match = re.search(r'<script id="scheme-payload" type="application/json">(.*?)</script>', document, re.S)
    assert match is not None
    payload = json.loads(match.group(1))["report"]["parameters"]
    assert payload["return_columns"] == ["P5L1", "P5L2", "P5L3"]
    assert payload["return_specs"] == report.parameters.return_specs
    assert {spec["periods"] for spec in payload["return_specs"].values()} == {5}
    assert {spec["price_column"] for spec in payload["return_specs"].values()} == {"open_hfq"}
    assert {spec["turnover_period"] for spec in payload["return_specs"].values()} == {5}
