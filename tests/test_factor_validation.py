"""Offline consistency checks for complete factor panels, without research services."""

from datetime import date, datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from scheme import Factor, FactorParams
from scheme.base.projects.factor import algo

START = date(2025, 1, 3)
END = date(2025, 1, 8)
DAYS = [START, date(2025, 1, 6), date(2025, 1, 7), END]
SYMBOLS = ["000001.XSHE", "600000.XSHG"]


def make_panel(days=DAYS):
    index = pd.MultiIndex.from_product([days, SYMBOLS], names=["date", "symbol"])
    return pd.DataFrame({
        "score": np.arange(len(index), dtype=float),
        "auxiliary": np.arange(len(index), dtype=float) * 10,
    }, index=index)


class OfflineFactor(Factor[FactorParams]):
    columns = ["score", "auxiliary"]

    def __init__(self, panel, change=None):
        super().__init__(FactorParams())
        self.panel = panel
        self.change = change
        self.calls = []

    def compute(self, start: date, end: date) -> pd.DataFrame:
        self.calls.append((start, end))
        if len(self.calls) == 1:
            return self.panel
        dates = pd.to_datetime(self.panel.index.get_level_values("date"))
        result = self.panel.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()
        return result if self.change is None else self.change(result)


@pytest.mark.parametrize("n_sample", [1, 2, 5])
def test_validation_calls_full_panel_once_and_each_unique_actual_sample_day(monkeypatch, n_sample):
    panel = make_panel()
    factor = OfflineFactor(panel)
    sample = Mock(side_effect=lambda population, count: population[-count:])
    log = Mock()
    monkeypatch.setattr(algo, "sample", sample)
    monkeypatch.setattr(algo.log, "info", log)

    assert factor.validate(START, END, n_sample=n_sample) is None
    expected = DAYS[-min(n_sample, len(DAYS)):]
    sample.assert_called_once_with(DAYS, min(n_sample, len(DAYS)))
    assert factor.calls == [(START, END), *((day, day) for day in expected)]
    assert len(set(factor.calls[1:])) == len(expected)
    assert all(day.weekday() < 5 for day, _ in factor.calls[1:])
    assert log.call_args.args[1] == ", ".join(day.isoformat() for day in expected)


def test_validation_default_samples_five_days_without_replacement():
    days = [day.date() for day in pd.bdate_range(START, periods=8)]
    factor = OfflineFactor(make_panel(days))
    assert factor.validate(days[0], days[-1]) is None
    assert factor.calls[0] == (days[0], days[-1])
    assert len(factor.calls[1:]) == len(set(factor.calls[1:])) == 5
    assert {day for day, _ in factor.calls[1:]} <= set(days)


def test_validation_single_day_is_a_regular_closed_interval():
    factor = OfflineFactor(make_panel([START]))
    assert factor.validate(START, START) is None
    assert factor.calls == [(START, START), (START, START)]


@pytest.mark.parametrize("n_sample", [0, -1, True, False, 1.5, "2", None, np.int64(2)])
def test_validation_requires_strict_positive_integer_before_computing(n_sample):
    factor = OfflineFactor(make_panel())
    with pytest.raises(ValueError, match="n_sample.*正整数"):
        factor.validate(START, END, n_sample=n_sample)
    assert factor.calls == []


def test_validation_rejects_reversed_interval_before_computing():
    factor = OfflineFactor(make_panel())
    with pytest.raises(ValueError, match="start <= end"):
        factor.validate(END, START)
    assert factor.calls == []


@pytest.mark.parametrize("timestamps", [False, True])
def test_validation_accepts_reordered_rows_columns_and_date_representations(timestamps):
    panel = make_panel([START])
    if timestamps:
        panel.index = pd.MultiIndex.from_product(
            [[pd.Timestamp(START)], SYMBOLS], names=["date", "symbol"],
        )

    def reorder(result):
        day = START if timestamps else pd.Timestamp(START)
        result.index = pd.MultiIndex.from_product([[day], SYMBOLS], names=["date", "symbol"])
        return result.iloc[::-1, ::-1]

    assert OfflineFactor(panel, reorder).validate(START, START) is None


def test_validation_accepts_numeric_tolerance_dtype_changes_and_equal_null_masks():
    panel = make_panel([START])
    panel["score"] = [1.0, 0.0]
    panel["auxiliary"] = pd.array([1, None], dtype="Int64")
    panel["label"] = ["up", None]
    panel["flag"] = [True, False]
    panel["category"] = pd.Categorical(["up", "down"], categories=["up", "down", "unused"])
    panel["object"] = pd.Series([2, 3], index=panel.index, dtype=object)

    def close_values(result):
        result["score"] = [1.0 + 0.5e-7, 0.5e-10]
        result["auxiliary"] = [1.0, np.nan]
        result["label"] = ["up", pd.NA]
        result["category"] = pd.Categorical(["up", "down"], categories=["down", "up"])
        result["object"] = [2, 3]
        return result

    class MixedFactor(OfflineFactor):
        columns = ["score", "auxiliary", "label", "flag", "category", "object"]

    assert MixedFactor(panel, close_values).validate(START, START) is None


@pytest.mark.parametrize("object_side", ["full", "single", "both"])
@pytest.mark.parametrize("difference,passes", [(0.5e-7, True), (2e-7, False)])
def test_object_numeric_columns_use_symmetric_tolerance(object_side, difference, passes):
    panel = make_panel([START])
    panel["score"] = pd.Series(
        [1.0, pd.NA] if object_side in {"full", "both"} else [1.0, np.nan],
        index=panel.index, dtype=object if object_side in {"full", "both"} else float,
    )

    def change(result):
        result["score"] = pd.Series(
            [1.0 + difference, pd.NA] if object_side in {"single", "both"}
            else [1.0 + difference, np.nan],
            index=result.index, dtype=object if object_side in {"single", "both"} else float,
        )
        return result

    factor = OfflineFactor(panel, change)
    if passes:
        assert factor.validate(START, START) is None
    else:
        with pytest.raises(ValueError, match=f"(?s){START}.*score"):
            factor.validate(START, START)


def test_numeric_strings_are_not_coerced_into_numbers():
    panel = make_panel([START])
    panel["score"] = ["1.0", "2.0"]

    def change(result):
        result["score"] = [1.0, 2.0]
        return result

    with pytest.raises(ValueError, match=f"(?s){START}.*score"):
        OfflineFactor(panel, change).validate(START, START)


@pytest.mark.parametrize("column,value,detail", [
    ("score", 1e-8, "score"),
    ("auxiliary", 2.0, "auxiliary"),
    ("label", "different", "label"),
    ("flag", False, "flag"),
])
def test_validation_compares_every_numeric_and_nonnumeric_column(column, value, detail):
    panel = make_panel([START])
    panel["label"] = ["up", "down"]
    panel["flag"] = [True, False]

    def change(result):
        result.loc[(START, SYMBOLS[0]), column] = value
        return result

    class MixedFactor(OfflineFactor):
        columns = ["score", "auxiliary", "label", "flag"]

    with pytest.raises(ValueError, match=f"(?s){START}.*{detail}"):
        MixedFactor(panel, change).validate(START, START)


@pytest.mark.parametrize("expected,actual", [(np.nan, 0.0), (0.0, np.nan), (np.nan, 1e-12)])
def test_validation_null_masks_cannot_be_hidden_by_near_zero_tolerance(expected, actual):
    panel = make_panel([START])
    panel.iloc[0, 0] = expected

    def change(result):
        result.iloc[0, 0] = actual
        return result

    with pytest.raises(ValueError, match=f"(?s){START}.*NULL.*score"):
        OfflineFactor(panel, change).validate(START, START)


@pytest.mark.parametrize("change,detail", [
    (lambda frame: frame.iloc[:1], "索引.*缺少.*600000"),
    (lambda frame: frame.rename(index={SYMBOLS[1]: "OTHER"}), "索引.*缺少.*额外.*OTHER"),
    (lambda frame: pd.concat([
        frame, pd.DataFrame({"score": [3.0], "auxiliary": [4.0]},
                            index=pd.MultiIndex.from_tuples([(START, "OTHER")],
                                                            names=["date", "symbol"])),
    ]), "索引.*额外.*OTHER"),
    (lambda frame: frame.drop(columns="auxiliary"), "列.*缺少.*auxiliary"),
    (lambda frame: frame.assign(extra=1.0), "列.*额外.*extra"),
])
def test_validation_rejects_missing_extra_securities_or_columns(change, detail):
    with pytest.raises(ValueError, match=f"(?s){START}.*{detail}"):
        OfflineFactor(make_panel([START]), change).validate(START, START)


@pytest.mark.parametrize("stage", ["full", "single"])
@pytest.mark.parametrize("kind,detail", [
    ("empty", "不能为空"), ("no_columns", "没有因子列"),
    ("duplicate_index", "唯一.*date, symbol"), ("duplicate_columns", "列不能重复"),
    ("index_names", "date, symbol"), ("one_level", "date, symbol"),
    ("extra_level", "date, symbol"), ("null_symbol", "非空值"),
    ("null_date", "非空值"), ("string_date", "date.*日期"),
    ("intraday", "日历日期"), ("not_frame", "DataFrame"),
])
def test_invalid_panels_never_pass_vacuously(stage, kind, detail):
    valid = make_panel([START])
    invalid = valid.copy()
    if kind == "empty":
        invalid = invalid.iloc[:0]
    elif kind == "no_columns":
        invalid = invalid.iloc[:, :0]
    elif kind == "duplicate_index":
        invalid = pd.concat([invalid, invalid.iloc[:1]])
    elif kind == "duplicate_columns":
        invalid.columns = ["score", "score"]
    elif kind == "index_names":
        invalid.index = invalid.index.set_names(["symbol", "date"])
    elif kind == "one_level":
        invalid.index = pd.Index(SYMBOLS, name="symbol")
    elif kind == "extra_level":
        invalid["extra"] = 1
        invalid = invalid.set_index("extra", append=True)
    elif kind in {"null_symbol", "null_date", "string_date", "intraday"}:
        day = {
            "null_date": pd.NaT, "string_date": START.isoformat(),
            "intraday": datetime(2025, 1, 3, 1),
        }.get(kind, START)
        symbols = [SYMBOLS[0], None] if kind == "null_symbol" else SYMBOLS
        invalid.index = pd.MultiIndex.from_product([[day], symbols], names=["date", "symbol"])
    elif kind == "not_frame":
        invalid = None
    factor = OfflineFactor(invalid if stage == "full" else valid, lambda _: invalid)
    context = "全区间" if stage == "full" else f"样本日期 {START}"
    with pytest.raises(ValueError, match=f"(?s){context}.*{detail}"):
        factor.validate(START, START)
    assert len(factor.calls) == (1 if stage == "full" else 2)


@pytest.mark.parametrize("day", [date(2025, 1, 2), date(2025, 1, 9)])
def test_validation_rejects_full_result_dates_outside_the_closed_interval(day):
    factor = OfflineFactor(make_panel([day]))
    with pytest.raises(ValueError, match="全区间结果日期超出"):
        factor.validate(START, END)
    assert factor.calls == [(START, END)]


@pytest.mark.parametrize("include_expected_day", [False, True])
def test_validation_does_not_clip_non_sample_dates_from_single_day_output(include_expected_day):
    wrong = make_panel([END])

    def change(frame):
        return pd.concat([frame, wrong]) if include_expected_day else wrong

    with pytest.raises(ValueError, match=f"(?s){START}.*非当天日期.*{END}"):
        OfflineFactor(make_panel([START]), change).validate(START, START)


@pytest.mark.parametrize("changed_day", [START, END])
def test_validation_snapshots_all_sampled_dates_before_compute_reuses_frame(monkeypatch, changed_day):
    monkeypatch.setattr(algo, "sample", lambda population, count: population[:count])

    class ReusingFactor(Factor[FactorParams]):
        columns = ["score", "auxiliary"]

        def __init__(self):
            super().__init__(FactorParams())
            self.panel = make_panel([START, END])
            self.calls = []

        def compute(self, start, end):
            self.calls.append((start, end))
            if len(self.calls) == 1:
                return self.panel
            dates = self.panel.index.get_level_values("date")
            # Even changing a later sample must not rewrite that sample's baseline.
            self.panel.loc[dates == changed_day, "score"] = 999.0
            self.panel.drop(index=self.panel.index[dates != start], inplace=True)
            if self.panel.empty:
                fresh = make_panel([start])
                fresh["score"] = 999.0
                for key, values in fresh.iterrows():
                    self.panel.loc[key, :] = values
            return self.panel

    factor = ReusingFactor()
    with pytest.raises(ValueError, match=f"(?s){changed_day}.*score"):
        factor.validate(START, END)


def test_validation_only_copies_sampled_date_rows(monkeypatch):
    panel = make_panel()
    copied_rows = []
    copy = pd.DataFrame.copy

    def track_copy(frame, deep=True):
        if deep:
            copied_rows.append(len(frame))
        return copy(frame, deep=deep)

    monkeypatch.setattr(pd.DataFrame, "copy", track_copy)
    factor = OfflineFactor(panel)
    assert factor.validate(START, END, n_sample=1) is None
    assert copied_rows
    assert max(copied_rows) <= len(SYMBOLS)


def test_validation_catches_late_day_values_depending_on_full_interval_start(monkeypatch):
    monkeypatch.setattr(algo, "sample", lambda population, count: population[-count:])

    class IntervalDependentFactor(Factor[FactorParams]):
        columns = ["score", "auxiliary"]

        def compute(self, start, end):
            panel = make_panel([day for day in DAYS if start <= day <= end])
            panel["score"] = panel.groupby(level="symbol")["score"].cumsum()
            return panel

    factor = IntervalDependentFactor(FactorParams())
    with pytest.raises(ValueError, match=f"(?s){END}.*score"):
        factor.validate(START, END, n_sample=1)


@pytest.mark.parametrize("stage", ["full", "single"])
def test_validation_preserves_compute_exceptions(stage):
    error = RuntimeError("offline compute failure")
    factor = OfflineFactor(make_panel([START]))
    factor.compute = Mock(side_effect=[error] if stage == "full" else [factor.panel, error])
    with pytest.raises(RuntimeError) as caught:
        factor.validate(START, START)
    assert caught.value is error
