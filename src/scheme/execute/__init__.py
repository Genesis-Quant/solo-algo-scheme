"""因子分析、策略组装和回测的执行入口。"""

from .factor import FactorAnalysisResult, analyze_factors
from .strategy import AlgoComponents, BacktestResult, ResearchBacktest, Strategy, run_backtest

__all__ = [
    "AlgoComponents", "BacktestResult", "FactorAnalysisResult", "ResearchBacktest",
    "Strategy", "analyze_factors", "run_backtest",
]
