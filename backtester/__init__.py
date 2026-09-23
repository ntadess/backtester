"""Event-driven backtester with a vectorized fast path and parity checks."""
from . import synthetic
from .costs import CostModel
from .data import BarDataHandler, align_bars, load_csv, load_yfinance
from .engine import EventDrivenBacktest
from .results import BacktestResult, metrics_table
from .strategies import MovingAverageCrossover, PairsTrading, Strategy, ZScoreMeanReversion
from .vectorized import run_vectorized

__all__ = [
    "synthetic", "CostModel", "BarDataHandler", "align_bars", "load_csv", "load_yfinance",
    "EventDrivenBacktest", "BacktestResult", "metrics_table", "run_vectorized",
    "Strategy", "MovingAverageCrossover", "ZScoreMeanReversion", "PairsTrading",
]
__version__ = "0.1.0"
