"""Real queries must preserve securities missing a close on the requested day."""

import os
from datetime import date

import pandas as pd
import pytest

from scheme import StockPool, Universe
from scheme.data.dolphindb import api
from scheme.data.dolphindb.symbols import engine_symbol

pytestmark = pytest.mark.skipif(
    os.getenv("SOLO_TEST_DOLPHIN") != "1", reason="需要显式开启真实 DolphinDB 测试",
)


def test_real_full_and_single_day_share_codes_without_that_days_quote(monkeypatch):
    monkeypatch.setattr(api, "_ALL_CODES", None)
    universe = Universe(pool=StockPool.CSI300)
    outputs = []
    snapshot = None
    for start, end in (
        (date(2026, 2, 5), date(2026, 2, 11)),
        (date(2026, 2, 9), date(2026, 2, 10)),
    ):
        with universe.query(start, end, factors=["close"]) as result:
            frame = result.data
        assert frame.code.nunique() == len(api._ALL_CODES)
        assert len(frame) == frame.time.nunique() * len(api._ALL_CODES)
        if snapshot is None:
            snapshot = api._ALL_CODES
        else:
            assert api._ALL_CODES is snapshot
        outputs.append(frame)
    assert "300442.SZ" in snapshot
    assert len(snapshot) > 4000
    full, daily = outputs
    full = full.loc[pd.to_datetime(full.time).eq(pd.Timestamp("2026-02-09"))]
    pd.testing.assert_frame_equal(
        full.sort_values("code").reset_index(drop=True),
        daily.sort_values("code").reset_index(drop=True),
    )
    missing_quote = daily.loc[daily.code.map(engine_symbol).eq("300442.XSHE")]
    assert len(missing_quote) == 1
    assert missing_quote[Universe.member].eq(True).all()
    assert missing_quote.close.isna().all()
