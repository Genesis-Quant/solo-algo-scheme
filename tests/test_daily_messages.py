from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scheme.base import StrategyAnalysisParams
from scheme.execute.strategy import api
from scheme.execute.strategy.api import ResearchBacktest

COLUMNS = ["open", "close", "pre_close", "up_limit", "down_limit"]


class Session:
    def __init__(self) -> None:
        self.uploads: dict[str, pd.DataFrame] = {}
        self.scripts: list[str] = []

    def upload(self, values: dict[str, pd.DataFrame]) -> None:
        self.uploads.update(values)

    def run(self, script: str) -> None:
        self.scripts.append(script)


def load(monkeypatch, rows: list[dict]) -> Session:
    frame = pd.DataFrame(rows, columns=["time", "code", *COLUMNS])
    frame["time"] = pd.to_datetime(frame.time)
    monkeypatch.setattr(api, "query", lambda _: nullcontext(SimpleNamespace(data=frame)))
    engine = object.__new__(ResearchBacktest)
    engine.session = Session()
    engine.parameters = StrategyAnalysisParams(
        start="2026-09-15", end="2026-09-16",
        symbols=sorted({row["code"] for row in rows}),
    )
    engine.load_messages(np.datetime64("2026-09-15"), np.datetime64("2026-09-16"))
    return engine.session


def row(code: str, **values: float | None) -> dict:
    prices = {"open": 10.0, "close": 10.5, "pre_close": 10.0, "up_limit": 11.0, "down_limit": 9.0}
    return {"time": "2026-09-15", "code": code, **prices, **values}


def test_daily_messages_skip_suspended_and_unlisted_securities(monkeypatch):
    session = load(monkeypatch, [
        row("600000.SH"),
        row("601059.SH", open=None, close=None, pre_close=None),
        row("001280.SZ", open=None, close=None, pre_close=None, up_limit=None, down_limit=None),
    ])
    daily = session.uploads["soloDaily"]
    assert daily.code.tolist() == ["600000.XSHG"]
    assert daily[COLUMNS].notna().all().all()
    assert session.scripts == ["soloMessages = soloDailyMessages(soloDaily)"]


def test_daily_messages_without_tradable_rows_are_empty(monkeypatch):
    session = load(monkeypatch, [row("601059.SH", open=None, close=None, pre_close=None)])
    assert "soloDaily" not in session.uploads
    assert session.scripts == ["soloMessages = table(array(TIMESTAMP,0) as timestamp)"]


@pytest.mark.parametrize("values", [
    {"up_limit": None}, {"down_limit": None}, {"pre_close": None}, {"open": None},
    {"close": None}, {"open": 0.0}, {"down_limit": -1.0},
])
def test_daily_bars_with_missing_or_invalid_prices_still_fail(monkeypatch, values):
    with pytest.raises(ValueError, match="不推测缺失价格"):
        load(monkeypatch, [row("600000.SH"), row("000001.SZ", **values)])
