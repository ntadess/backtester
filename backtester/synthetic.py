"""Synthetic price generators with known statistical properties.

Useful for known-answer tests: if we generate a cointegrated pair, a pairs
strategy *should* make money; if it doesn't, the engine or strategy is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Bars


def _index(n: int, start: str, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq)


def bars_from_close(
    close: np.ndarray,
    index: pd.Index,
    rng: np.random.Generator,
    gap_vol: float = 0.002,
    range_vol: float = 0.004,
) -> pd.DataFrame:
    """Build internally consistent OHLCV bars around a close series."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    prev_close = np.concatenate(([close[0]], close[:-1]))
    open_ = prev_close * np.exp(rng.normal(0.0, gap_vol, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, range_vol, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, range_vol, n)))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def gbm(n: int = 2520, s0: float = 100.0, mu: float = 0.07, sigma: float = 0.2,
        seed: int = 0, symbol: str = "GBM", periods_per_year: int = 252,
        start: str = "2015-01-01", freq: str = "B") -> Bars:
    """Geometric Brownian motion: a random walk with no exploitable edge."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = s0 * np.exp(np.cumsum(log_ret))
    return {symbol: bars_from_close(close, _index(n, start, freq), rng)}


def trending(n: int = 2520, s0: float = 100.0, drift: float = 0.5, sigma: float = 0.2,
             switch_prob: float = 0.01, seed: int = 0, symbol: str = "TREND",
             periods_per_year: int = 252, start: str = "2015-01-01", freq: str = "B") -> Bars:
    """GBM whose drift flips between +drift and -drift (persistent regimes).

    Regimes last 1/switch_prob bars on average, so trend-following has an edge.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    flips = rng.random(n) < switch_prob
    regime = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)
    log_ret = (regime * drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = s0 * np.exp(np.cumsum(log_ret))
    return {symbol: bars_from_close(close, _index(n, start, freq), rng)}


def mean_reverting(n: int = 2520, level: float = 100.0, half_life: float = 10.0,
                   sigma: float = 0.01, seed: int = 0, symbol: str = "OU",
                   start: str = "2015-01-01", freq: str = "B") -> Bars:
    """Ornstein-Uhlenbeck process in log price, pulled back toward `level`."""
    rng = np.random.default_rng(seed)
    theta = 1.0 - 0.5 ** (1.0 / half_life)
    mu = np.log(level)
    x = np.empty(n)
    x[0] = mu
    shocks = sigma * rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + shocks[t]
    return {symbol: bars_from_close(np.exp(x), _index(n, start, freq), rng)}


def cointegrated_pair(n: int = 2520, beta: float = 1.5, intercept: float = 10.0,
                      x0: float = 50.0, sigma_x: float = 0.015, spread_sd: float = 1.0,
                      half_life: float = 10.0, seed: int = 0,
                      symbols: tuple[str, str] = ("Y", "X"),
                      start: str = "2015-01-01", freq: str = "B") -> Bars:
    """Y = intercept + beta * X + stationary AR(1) spread; X is a random walk.

    Returns {symbols[0]: Y bars, symbols[1]: X bars}.
    """
    rng = np.random.default_rng(seed)
    x = x0 * np.exp(np.cumsum(sigma_x * rng.standard_normal(n)))
    phi = 0.5 ** (1.0 / half_life)
    eps = spread_sd * np.sqrt(1.0 - phi**2) * rng.standard_normal(n)
    spread = np.empty(n)
    spread[0] = eps[0]
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + eps[t]
    y = intercept + beta * x + spread
    idx = _index(n, start, freq)
    y_sym, x_sym = symbols
    return {y_sym: bars_from_close(y, idx, rng), x_sym: bars_from_close(x, idx, rng)}
