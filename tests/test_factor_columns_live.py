"""Real end-to-end factor column declarations and independent exposure queries."""

import os
from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scheme import Factor, FactorParams, StockPool, Universe

pytestmark = pytest.mark.skipif(
    os.getenv("SOLO_TEST_DOLPHIN") != "1", reason="需要显式开启真实 DolphinDB 测试",
)


class LocalFactor(Factor[FactorParams]):
    columns = ["local_score"]

    def compute(self, start, end):
        with self.params.universe.query(
            start, end + timedelta(days=1), filters=[Universe.member],
        ) as result:
            keys = result.data[["time", "code"]].sort_values(["time", "code"])
        frame = keys.assign(local_score=np.sin(np.arange(len(keys)) * 0.7))
        return self.preprocess(frame)


def test_real_locally_constructed_factor_queries_its_own_exposures():
    factor = LocalFactor(FactorParams(universe=Universe(pool=StockPool.CSI300)))
    result = factor.compute(date(2026, 2, 9), date(2026, 2, 9))
    assert list(result) == factor.columns
    assert len(result) >= 290
    assert result.local_score.notna().sum() >= 280
    assert abs(result.local_score.mean()) < 1e-10
    assert abs(result.local_score.std(ddof=0) - 1) < 1e-10
    assert pd.isna(result.loc[(pd.Timestamp("2026-02-09"), "300442.XSHE"), "local_score"])


def test_real_template_full_and_daily_results_keep_missing_quote_key(monkeypatch):
    package = Path(__file__).resolve().parents[2] / "factor/src/factor"
    if not package.is_dir():
        pytest.skip("Factor template is in the parent Algo repository")
    import sys

    name = "_solo_factor_live_template"
    spec = spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)],
    )
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    factor = module.Factor(module.FactorParams(universe=Universe(pool=StockPool.CSI300)))
    full = factor.compute(date(2026, 2, 5), date(2026, 2, 10))
    daily = factor.compute(date(2026, 2, 9), date(2026, 2, 9))
    assert list(full) == ["momentum"] == list(factor.columns)
    expected = full.loc[[pd.Timestamp("2026-02-09")]]
    pd.testing.assert_frame_equal(expected, daily, rtol=1e-7, atol=1e-10)
    key = (pd.Timestamp("2026-02-09"), "300442.XSHE")
    assert key in expected.index and key in daily.index
    assert pd.isna(expected.loc[key, "momentum"])
    factor.validate(date(2026, 2, 5), date(2026, 2, 10), n_sample=2)
    parameters = module.FactorReportForm(
        start="2026-02-05", end="2026-02-10", weight="equal", n_select=2,
        period=1, price_column="open_hfq", max_lag=1,
    ).build()
    assert parameters.columns == []
    report = parameters.run(module.Factor)
    assert report.parameters.columns == factor.columns
    assert "momentum_P1L1_ic" in report.information_coefficient
    assert "momentum_P1L1_top" in report.group_returns
