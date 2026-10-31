"""Factor DOS stays directly inside the existing execution scripts."""

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scheme.base.projects.factor import preprocess
from scheme.execute.factor.api import ANALYSIS_SCRIPT

PREPROCESS_SCRIPT = preprocess.PREPROCESS_SCRIPT


def test_factor_functions_are_literals_in_existing_files():
    import importlib

    modules = [
        (importlib.import_module("scheme.base.projects.factor.preprocess"), "PREPROCESS_SCRIPT", "factorPreprocess"),
        (importlib.import_module("scheme.execute.factor.api"), "ANALYSIS_SCRIPT", "factorGroupTurnover"),
    ]
    for module, name, function in modules:
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert isinstance(assignment.value, ast.Constant)
        script = ast.literal_eval(assignment.value)
        assert f"def {function}(" in script
        assert "use factor" not in script
        assert "factor::" not in script
    assert not Path(modules[0][0].__file__).with_name("dolphindb.py").exists()


@pytest.fixture(scope="module")
def dolphin():
    if os.getenv("QUERY_TEST_DOLPHINDB") != "1":
        pytest.skip("requires explicit opt-in to the configured DolphinDB server")
    from query import create_session

    session = create_session(max_time=300)
    try:
        session.run(PREPROCESS_SCRIPT)
        session.run(ANALYSIS_SCRIPT)
        yield session
    finally:
        session.close()


def test_inline_preprocessing_matches_numpy_svd(dolphin):
    raw = pd.DataFrame({
        "time": pd.to_datetime(["2025-01-06"] * 12),
        "code": [f"{index:06d}.SZ" for index in range(12)],
        "signal": [1.0, 4, 2, 6, 3, 9, 5, 10, 7, 11, 8, 12],
        "mktmv": np.exp(np.arange(12) / 12 + 5),
        "industry": [1, 2, 2, 3] * 3,
    })
    dolphin.upload({"inlineFactorRows": raw})
    actual = dolphin.run('factorPreprocess(inlineFactorRows,["signal"],3)').set_index("code").sort_index()
    dolphin.upload({"inlineFactorRows": raw.iloc[::-1].reset_index(drop=True)})
    permuted = dolphin.run('factorPreprocess(inlineFactorRows,["signal"],3)').set_index("code").sort_index()
    pd.testing.assert_frame_equal(actual, permuted)
    values = raw.signal.to_numpy()
    median = np.median(values)
    mad = np.median(abs(values - median))
    values = np.clip(values, median - 3 * mad, median + 3 * mad)
    y = (values - values.mean()) / values.std()
    mv = np.log(raw.mktmv.to_numpy())
    x = np.column_stack([np.ones(len(raw)), (mv - mv.mean()) / mv.std(), raw.industry == 2, raw.industry == 3])
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    np.testing.assert_allclose(actual.signal, (residual - residual.mean()) / residual.std(), atol=1e-12)


def test_inline_turnover_keeps_empty_cross_sections_on_calendar(dolphin):
    result = dolphin.run('''
        inlineFactorRows=table(timestamp(take(2025.01.06,2) join take(2025.01.08,2)) as time,
            take(`A`B,4) as code,1.0 2 1 2 as signal,0 1 0 1 as signal_group)
        factorGroupTurnover(inlineFactorRows,["signal"],1 2,2,1,"time","code",2025.01.06 2025.01.07 2025.01.08)
    ''').set_index(["time", "periods"])
    assert len(result) == 6
    assert result.loc[("2025-01-08", 1), "group0"] == 1
    assert result.loc[("2025-01-08", 2), "group0"] == 0


def test_inline_group_statistics_preserve_weighting(dolphin):
    dolphin.run('''
        inlineFactorRows=table(take(timestamp(2025.01.06),4) as time,`A`B`C`D as code,
            take(1.0,4) as signal,0 0 1 1 as signal_group,
            take(1.0,4) as mktmv,1.0 2 3 4 as ret)
    ''')
    result = dolphin.run('factorGroupReturns(inlineFactorRows,["ret"],["signal"],2,1)')
    assert result.signal_ret_bottom.iloc[0] == 1
    assert result.signal_ret_top.iloc[0] == 4
