"""optimize 项目的默认实现。"""

from __future__ import annotations

from datetime import timedelta
from numbers import Real
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.optimize import minimize

from scheme.base.internal.context import ResearchContext
from scheme.base.internal.types import Signal, Target
from scheme.base.projects.optimize import OptimizeAlgo
from scheme.base.projects.optimize import OptimizeParams as BaseOptimizeParams
from scheme.data import query
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol
from scheme.log import get_logger

__all__ = ["risk_parity", "OptimizeParams", "RiskParity"]

log = get_logger("scheme.optimize")

C = TypeVar("C", bound=ResearchContext[Any], infer_variance=True)


def risk_parity(returns: pd.DataFrame) -> tuple[np.ndarray, str | None]:
    count = len(returns.columns)
    if not count:
        return np.array([]), None
    equal = np.full(count, 1 / count)
    valid = returns.dropna()
    # Sample covariance needs at least two rows.
    if len(valid) < 2:
        return equal, "历史样本不足"
    covariance = valid.cov().to_numpy()
    covariance /= np.trace(covariance)

    def objective(weights: np.ndarray) -> float:
        contribution = weights * (covariance @ weights)
        return float(np.sum((contribution / contribution.sum() - 1 / count) ** 2))

    result = minimize(
        objective,
        equal,
        bounds=[(1e-8, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        return equal, "风险平价求解未收敛"
    return result.x, None


class OptimizeParams(BaseOptimizeParams):
    lookback_days: int = Field(default=180, ge=30)
    gross_exposure: float = Field(default=0.98, ge=0, le=1)


class RiskParity(OptimizeAlgo[OptimizeParams, C], Generic[C]):
    def __init__(self, params: OptimizeParams | None = None) -> None:
        super().__init__(params or OptimizeParams())

    def on_signal(self, signals: Signal[Any]) -> None:
        self.ctx.target = self.allocate(signals)

    def allocate(self, signals: Signal[Any]) -> Target:
        if any(not isinstance(s, Real) or not np.isfinite(s) for s in signals.values()):
            raise TypeError("默认风险平价接受有符号数值 Signal；其他值类型请提供自定义 Optimize")
        symbols = [s for s, value in signals.items() if value != 0]
        if not symbols:
            return {}
        today = pd.Timestamp(self.backtest.time).date()
        with query(
            {
                "start_date": (today - timedelta(days=self.params.lookback_days)).isoformat(),
                "end_date": (today - timedelta(days=1)).isoformat(),
                "codes": [source_symbol(s) for s in symbols],
                "factors": ["close", "adj_factor"],
            }
        ) as result:
            history = result.data
        history["code"] = history.code.map(engine_symbol)
        if history.empty or not {"close", "adj_factor"}.issubset(history.columns):
            returns = pd.DataFrame(columns=symbols, dtype=float)
        else:
            history["adjusted"] = history.close * history.adj_factor
            prices = history.pivot(index="time", columns="code", values="adjusted").reindex(
                columns=symbols
            )
            returns = prices.pct_change(fill_method=None).mul(
                [np.sign(signals[s]) for s in symbols]
            )
        weights, reason = risk_parity(returns)
        target = {
            s: float(np.sign(signals[s]) * self.params.gross_exposure * w)
            for s, w in zip(symbols, weights, strict=True)
        }
        self.diagnostics.append(
            {
                "time": pd.Timestamp(self.backtest.time),
                "method": "equal_weight" if reason else "risk_parity",
                "reason": reason,
            }
        )
        if reason:
            log.warning("风险平价退化为等权：{}（{} 只）", reason, len(symbols))
        else:
            log.debug(
                "风险平价分配 {} 只，总仓位 {:.0%}，最大权重 {:.2%}",
                len(symbols), self.params.gross_exposure,
                max((abs(value) for value in target.values()), default=0.0),
            )
        return target
