"""Strategy interface and shared indicator helpers.

Every strategy implements the same logic twice:

  generate_targets(bars)  vectorized, whole history at once (fast sweeps)
  on_bar(data)            incremental, sees only data up to the current bar

Both return *sparse target weights*: a new instruction only when the
strategy's state changes, NaN (vectorized) or None (event) otherwise.
The parity tests assert both paths produce identical fills, which catches
lookahead bugs in the vectorized code (e.g. a forgotten shift).

The indicator helpers below are built on sliding_window_view so the
vectorized path does the exact same floating-point arithmetic as calling
np.mean / np.std on the trailing window in the event path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..data import BarDataHandler, Bars

Weights = dict[str, float]


class Strategy(ABC):
    symbols: tuple[str, ...]

    def reset(self) -> None:
        """Clear incremental state before an event-driven run."""

    @abstractmethod
    def generate_targets(self, bars: Bars) -> pd.DataFrame:
        """Index = bar timestamps, columns = self.symbols, NaN = no new target."""

    @abstractmethod
    def on_bar(self, data: BarDataHandler) -> Weights | None:
        """Return new target weights, or None to keep current positions."""


# ---------------------------------------------------------------- indicators

def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= window:
        out[window - 1:] = sliding_window_view(a, window).mean(axis=1)
    return out


def rolling_std(a: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= window:
        out[window - 1:] = sliding_window_view(a, window).std(axis=1)
    return out


def rolling_zscore(a: np.ndarray, window: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (a - rolling_mean(a, window)) / rolling_std(a, window)
    z[~np.isfinite(z)] = np.nan
    return z


def zscore_last(window: np.ndarray) -> float:
    """z-score of the last element of `window` (event-path twin of rolling_zscore)."""
    sd = window.std()
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float((window[-1] - window.mean()) / sd)


def rolling_ols(y: np.ndarray, x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling OLS of y on x with intercept. Returns (beta, alpha)."""
    beta = np.full(len(y), np.nan)
    alpha = np.full(len(y), np.nan)
    if len(y) >= window:
        yw = sliding_window_view(y, window)
        xw = sliding_window_view(x, window)
        ym = yw.mean(axis=1)
        xm = xw.mean(axis=1)
        dx = xw - xm[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (dx * (yw - ym[:, None])).sum(axis=1) / (dx * dx).sum(axis=1)
        beta[window - 1:] = b
        alpha[window - 1:] = ym - b * xm
    return beta, alpha


def ols_last(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS on a single window (event-path twin of rolling_ols)."""
    ym = y.mean()
    xm = x.mean()
    dx = x - xm
    denom = (dx * dx).sum()
    if denom == 0:
        return np.nan, np.nan
    b = (dx * (y - ym)).sum() / denom
    return float(b), float(ym - b * xm)


# ------------------------------------------------------------- state helpers

def hysteresis_states(z: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    """Vectorized mean-reversion state machine.

    z > entry -> -1 (short), z < -entry -> +1 (long), |z| < exit -> 0,
    otherwise hold the previous state. Starts flat.
    """
    raw = np.full(len(z), np.nan)
    with np.errstate(invalid="ignore"):
        raw[np.abs(z) < exit_] = 0.0
        raw[z > entry] = -1.0
        raw[z < -entry] = 1.0
    return pd.Series(raw).ffill().fillna(0.0).to_numpy()


def hysteresis_step(z: float, entry: float, exit_: float, prev: float) -> float:
    """Single-step twin of hysteresis_states."""
    if np.isnan(z):
        return prev
    if z > entry:
        return -1.0
    if z < -entry:
        return 1.0
    if abs(z) < exit_:
        return 0.0
    return prev


def targets_from_states(
    index: pd.Index,
    symbols: tuple[str, ...],
    states: np.ndarray,
    weights_at: Callable[[int, float], Weights],
) -> pd.DataFrame:
    """Emit a target row only where the state changes (starting from flat)."""
    targets = pd.DataFrame(np.nan, index=index, columns=list(symbols))
    prev = np.concatenate(([0.0], states[:-1]))
    for i in np.flatnonzero(states != prev):  # loops over changes only
        w = weights_at(int(i), float(states[i]))
        targets.iloc[i] = [w[s] for s in symbols]
    return targets
