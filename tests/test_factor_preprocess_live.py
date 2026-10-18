"""Real DOS lifecycle and NumPy-oracle regression with independently queried exposures."""

import os
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from backtest import DosVar

from scheme import Factor, FactorParams
from scheme.config import DolphinSettings
from scheme.data.dolphindb.database import create_session
from scheme.data.dolphindb.symbols import engine_symbol

pytestmark = pytest.mark.skipif(
    os.getenv("SOLO_TEST_DOLPHIN") != "1", reason="需要显式开启真实 DolphinDB 测试",
)
module = import_module("scheme.base.projects.factor.preprocess")


class PreprocessedFactor(Factor[FactorParams]):
    columns = ["alpha", "currentDate", "integer", "mad_zero", "explained", "constant"]

    def compute(self, start, end):
        raise NotImplementedError


def synthetic_factors():
    frames = []
    exposures = []
    for index, (day, count) in enumerate((
        ("2025-01-03", 64), ("2025-01-06", 64), ("2025-01-07", 3),
        ("2025-01-08", 8), ("2025-01-09", 24),
    )):
        values = np.arange(count, dtype=float)
        industry = np.asarray(["bank", "energy", "tech"], dtype=object)[np.arange(count) % 3]
        frame = pd.DataFrame({
            "time": pd.to_datetime([day] * count),
            "code": [f"{code:06d}.SZ" for code in range(1, count + 1)],
            "alpha": values * 0.2 + np.sin(values * 0.7 + index),
            "currentDate": np.cos(values * 0.4 + index),
            "integer": (values.astype(int) * 13) % 23,
            "mad_zero": np.where(values < count * 0.75, 0.0, values + 1),
            "explained": (industry != "bank").astype(float),
            "constant": np.repeat(7, count),
        })
        market_value = np.exp(2 + values / 25)
        if index < 2:
            market_value[:4] = [-5.0, 0.0, 0.5, np.nan]
            industry[4] = None
            frame.loc[5, "alpha"] = np.nan
            frame.loc[count - 1, "alpha"] = 1e6  # Positive MAD must clip this outlier.
        elif index == 3:
            market_value = np.resize([-5.0, 0.0, 0.5, -1.0], count)
        elif index == 4:
            # Market value is collinear with the industry dummies, but n > effective rank.
            market_value = np.exp(2 + np.arange(count) % 3)
        exposure = frame[["time", "code"]].assign(circ_mv=market_value, industry_l0=industry)
        frames.append(frame)
        exposures.append(exposure)
    # Pure local factors contain no query intermediates, membership or exposures.
    return (
        pd.concat(frames, ignore_index=True).sample(frac=1, random_state=17).reset_index(drop=True),
        pd.concat(exposures, ignore_index=True).sample(frac=1, random_state=23).reset_index(drop=True),
    )


def numpy_preprocess(frame, exposures, columns):
    """Independent test oracle: raw MAD, population zscores and intercept-first SVD OLS."""
    joined = frame.merge(exposures, on=["time", "code"], how="left", validate="one_to_one")
    joined = joined.sort_values(["time", "code"]).reset_index(drop=True)
    index = pd.MultiIndex.from_arrays(
        [joined.time, joined.code.map(engine_symbol)], names=["date", "symbol"],
    )
    expected = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    designs = {}
    eps = np.finfo(float).eps
    for day, group in joined.groupby("time", sort=True):
        for column in columns:
            raw = group[column].to_numpy(dtype=float)
            market_value = group.circ_mv.to_numpy(dtype=float)
            valid = np.isfinite(raw) & np.isfinite(market_value) & group.industry_l0.notna().to_numpy()
            if not valid.any():
                continue
            mv = np.log(np.maximum(market_value[valid], 1.0))
            mv_std = mv.std(ddof=0)
            if not np.isfinite(mv_std) or mv_std <= 0:
                continue
            mv = (mv - mv.mean()) / mv_std
            y = raw[valid]
            median = np.median(y)
            mad = np.median(np.abs(y - median))
            if np.isfinite(mad) and mad > 0:
                y = np.clip(y, median - 3 * mad, median + 3 * mad)
            y_std = y.std(ddof=0)
            if not np.isfinite(y_std) or y_std <= 0:
                continue
            y = (y - y.mean()) / y_std
            regression = np.isfinite(y) & np.isfinite(mv)
            y, mv = y[regression], mv[regression]
            if not len(y):
                continue
            industries = group.industry_l0.to_numpy()[valid][regression].astype(str)
            levels = np.sort(np.unique(industries))
            design = np.column_stack([
                np.ones(len(y)), mv,
                *[(industries == industry).astype(float) for industry in levels[1:]],
            ])
            singular = np.linalg.svd(design, compute_uv=False)
            rank_tolerance = max(design.shape) * singular.max() * eps
            if len(y) <= np.count_nonzero(singular > rank_tolerance):
                continue
            beta = np.linalg.lstsq(design, y, rcond=max(design.shape) * eps)[0]
            residual = y - design @ beta
            tolerance = np.max(np.abs(y)) * max(design.shape) * eps
            residual_std = residual.std(ddof=0)
            if not np.isfinite(residual_std) or residual_std <= tolerance:
                continue
            residual = (residual - residual.mean()) / residual_std
            keys = index[group.index.to_numpy()[valid][regression]]
            expected.loc[keys, column] = residual
            designs[(day, column)] = (keys, design)
    return expected.sort_index(), designs


@pytest.mark.parametrize("use_variable", [False, True])
def test_real_preprocess_returns_and_cleans_on_success_and_failure(monkeypatch, use_variable):
    frame, exposures = synthetic_factors()
    original = frame.copy(deep=True)
    factor = PreprocessedFactor(FactorParams())
    expected, designs = numpy_preprocess(frame, exposures, factor.columns)
    prefix = "soloCleanupTest" + uuid4().hex
    source = prefix + "Source"
    sentinel = prefix + "Sentinel"
    missing = prefix + "Missing"
    query_sessions = []
    settings = DolphinSettings.from_env()
    with create_session(settings) as session:
        def query(request, *, session):
            # Only query data is synthetic. Its separate owned DDB session is real.
            assert session is not caller
            assert session.run("1 + 1") == 2
            assert sentinel not in session.run("objs().name")
            assert request == {
                "start_date": "2025-01-03", "end_date": "2025-01-09",
                "codes": sorted(frame.code.unique().tolist()),
                "factors": ["circ_mv", "industry_l0"],
            }
            query_sessions.append(session)
            return SimpleNamespace(data=exposures.copy(deep=True))

        caller = session
        monkeypatch.setattr(module, "query", query)
        session.upload({source: frame, sentinel: 42})
        session.run(f'if ("{missing}" in objs().name) {{ undef("{missing}", VAR) }}')
        previous = None
        for attempt in range(2):
            result = factor.preprocess(
                DosVar(source) if use_variable else frame, session=session, settings=settings,
            )
            assert len(query_sessions) == attempt + 1
            assert len(result) == len(frame) and list(result) == factor.columns
            assert result.index.names == ["date", "symbol"]
            assert result.index.is_unique and result.index.is_monotonic_increasing
            assert all(pd.api.types.is_float_dtype(dtype) for dtype in result.dtypes)
            pd.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-8, atol=1e-9)
            assert result.constant.isna().all() and result.explained.isna().all()
            for day in ("2025-01-07", "2025-01-08"):
                assert result.loc[pd.Timestamp(day)].isna().all().all()
            assert result.loc[pd.Timestamp("2025-01-09"), "alpha"].notna().all()
            for day in ("2025-01-03", "2025-01-06"):
                raw_mad_zero = frame.loc[frame.time == day, "mad_zero"]
                assert np.median(np.abs(raw_mad_zero - np.median(raw_mad_zero))) == 0
                assert raw_mad_zero.nunique() > 1
                assert result.loc[pd.Timestamp(day), "mad_zero"].notna().any()
                integer = result.loc[pd.Timestamp(day), "integer"].dropna()
                assert not np.allclose(integer, np.round(integer))
            for (day, column), (keys, design) in designs.items():
                residual = result.loc[keys, column].to_numpy()
                assert abs(residual.mean()) < 1e-10
                assert abs(residual.std(ddof=0) - 1) < 1e-10
                # Captures a wrong beta order even if residuals are restandardized.
                np.testing.assert_allclose(design.T @ residual / len(residual), 0.0, atol=1e-9)
            if previous is not None:
                pd.testing.assert_frame_equal(result, previous)
            previous = result
            assert not any(str(name).startswith("soloFp") for name in session.run("objs().name"))
            assert session.run(sentinel) == 42
            assert session.run(f"{source}.rows()") == len(frame)
        stored = session.run(f"select * from {source}")
        assert list(stored) == list(original)
        assert stored.code.tolist() == original.code.tolist()
        assert pd.to_datetime(stored.time).tolist() == original.time.tolist()
        for column in factor.columns:
            np.testing.assert_array_equal(stored[column], original[column])
        pd.testing.assert_frame_equal(frame, original)
        all_null = frame.copy()
        all_null[factor.columns] = np.nan
        if not use_variable:
            with pytest.raises(ValueError, match="all NULL"):
                factor.preprocess(all_null, session=session, settings=settings)
        # Both cases also exercise an actual server-side failure and finally cleanup.
        session.upload({source: all_null})
        with pytest.raises(RuntimeError, match="all NULL"):
            factor.preprocess(DosVar(source), session=session, settings=settings)
        assert len(query_sessions) == 3 and len({id(item) for item in query_sessions}) == 3
        assert not any(str(name).startswith("soloFp") for name in session.run("objs().name"))
        assert session.run(sentinel) == 42
        assert session.run("1 + 1") == 2
