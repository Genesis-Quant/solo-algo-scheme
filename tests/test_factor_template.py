import sys
from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

from scheme import Universe


@pytest.fixture
def template(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "factor/src/factor"
    if not directory.is_dir():
        pytest.skip("Factor template is in the parent Algo repository")

    package = "_solo_factor_template_test"
    spec = spec_from_file_location(
        package, directory / "__init__.py", submodule_search_locations=[str(directory)],
    )
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package, module)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in (f"{package}.factor", f"{package}.params"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("start,end", [
    (date(2025, 1, 6), date(2025, 1, 6)),
    (date(2025, 1, 3), date(2025, 1, 6)),
])
def test_template_closed_interval_uses_half_open_universe_query(template, monkeypatch, start, end):
    calls = []

    class Result:
        def __init__(self, first, last):
            days = pd.bdate_range(first, last, inclusive="left")
            self.session = object()
            self.data_ref = "templateFactorData"
            self.data = pd.DataFrame({
                "time": days,
                "code": ["000001.SZ"] * len(days),
                "momentum": [day.day / 100 for day in days],
            })

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    current = {}
    preprocessing = []

    def query(self, first, last, **kwargs):
        calls.append((first, last, kwargs))
        current["result"] = Result(first, last)
        return current["result"]

    def preprocess(self, data, *, session):
        result = current["result"]
        assert data == result.data_ref
        assert session is result.session
        assert self.columns == ["momentum"]
        preprocessing.append(data)
        frame = result.data[["time", "code", *self.columns]].copy()
        frame = frame.rename(columns={"time": "date", "code": "symbol"})
        frame["symbol"] = frame["symbol"].str.replace(".SZ", ".XSHE", regex=False)
        return frame.set_index(["date", "symbol"]).sort_index()

    monkeypatch.setattr(Universe, "query", query)
    monkeypatch.setattr(template.Factor, "preprocess", preprocess)
    params = template.FactorParams()
    assert "start" not in type(params).model_fields
    assert "end" not in type(params).model_fields
    factor = template.Factor(params)
    panel = factor.compute(start, end)
    assert calls[0][0:2] == (start, end + timedelta(days=1))
    assert calls[0][2]["lookback"] == timedelta(days=110)
    assert calls[0][2]["filters"] == [Universe.member]
    assert "factors" not in calls[0][2]
    assert set(calls[0][2]["derivatives"]) == {"adjusted_close", "momentum"}
    assert template.Factor.__dict__["columns"] == ["momentum"]
    assert "columns" not in template.FactorAnalysisParams.__annotations__
    assert not template.FactorAnalysisParams.model_fields["columns"].is_required()
    assert template.FactorAnalysisParams(start=start, end=end).columns == []
    assert "columns" not in template.FactorReportForm.model_fields
    assert preprocessing == ["templateFactorData"]
    assert panel.index.get_level_values("date").date.tolist() == pd.bdate_range(start, end).date.tolist()
    assert set(panel.index.get_level_values("symbol")) == {"000001.XSHE"}

    calls.clear()
    assert factor.validate(start, end, n_sample=5) is None
    assert calls[0][0:2] == (start, end + timedelta(days=1))
    assert len(calls) == 1 + len(panel)
    for first, last, _ in calls[1:]:
        assert last == first + timedelta(days=1)


def test_template_rejects_reversed_dates_before_query(template, monkeypatch):
    def query(*args, **kwargs):
        pytest.fail("Reversed dates must not query data")

    monkeypatch.setattr(Universe, "query", query)
    with pytest.raises(ValueError, match="start <= end"):
        template.Factor(template.FactorParams()).compute(date(2025, 1, 7), date(2025, 1, 6))


def test_factor_form_labels_end_as_inclusive_and_builds_single_day(template):
    form = template.FactorReportForm(start="2025-01-06", end="2025-01-06")
    assert form.model_json_schema()["properties"]["end"]["title"] == "结束日期（含）"
    analysis = form.build()
    assert analysis.start == analysis.end == date(2025, 1, 6)
