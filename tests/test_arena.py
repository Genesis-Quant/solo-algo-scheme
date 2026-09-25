"""直接导出的 Arena 接口与真实上下文生命周期验证。"""

import os
from uuid import uuid4

import numpy as np
import pytest
from runtime.apps.backtest import run_backtest
from runtime.apps.backtest.schema import CALLBACK_PARAMETER_COUNTS
from runtime.apps.query import execute_query

from scheme.data import backtest, query


def test_exports_are_arena_functions():
    assert query is execute_query
    assert backtest is run_backtest


@pytest.mark.skipif(os.getenv("SOLO_TEST_DOLPHIN") != "1", reason="需要 DolphinDB")
def test_query_and_backtest_with():
    request = {
        "start_date": "2026-06-01",
        "end_date": "2026-06-02",
        "codes": ["000001.SZ", "600000.SH"],
        "factors": ["open", "close"],
        "derivatives": {
            "ratio": {
                "type": "DIRECT",
                "op": "binary.div",
                "fields": {"left": "close", "right": "open"},
            },
        },
    }
    with pytest.raises(ValueError, match="close-on-error"):
        with query(request) as result:
            frame = result.data
            assert len(frame) == 4
            np.testing.assert_allclose(frame.ratio, frame.close / frame.open)
            raise ValueError("close-on-error")
    assert result.closed
    with pytest.raises(RuntimeError, match="关闭"):
        _ = result.data

    callbacks = {}
    for name, count in CALLBACK_PARAMETER_COUNTS.items():
        arguments = ["mutable context", "message", "indicator"][:count]
        callbacks[name] = f"def {name}({', '.join(arguments)}) {{ return NULL }}"
    callbacks["onSnapshot"] = """
        def onSnapshot(mutable context, message, indicator) {
            for (code in message.symbol)
                backtest::order_target(context, message, code, 100l, "with-check")
        }
    """
    with backtest(
        request,
        callbacks,
        name="soloArena" + uuid4().hex,
        config={"cash": 100000},
    ) as report:
        assert len(report.daily_portfolios) == 2
        assert not report.trade_details.empty
    assert report.closed
    with pytest.raises(RuntimeError, match="关闭"):
        _ = report.daily_portfolios
