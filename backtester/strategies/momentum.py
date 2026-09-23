from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import BarDataHandler, Bars
from .base import Strategy, Weights, rolling_mean, targets_from_states


class MovingAverageCrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, short (or flat) below."""

    def __init__(self, symbol: str, fast: int = 20, slow: int = 100, long_only: bool = False):
        if not 0 < fast < slow:
            raise ValueError("need 0 < fast < slow")
        self.symbol, self.fast, self.slow, self.long_only = symbol, fast, slow, long_only
        self.symbols = (symbol,)
        self.reset()

    def reset(self) -> None:
        self._state = 0.0

    def _clip(self, s: float | np.ndarray):
        return np.maximum(s, 0.0) if self.long_only else s

    def generate_targets(self, bars: Bars) -> pd.DataFrame:
        close = bars[self.symbol]["close"].to_numpy()
        diff = rolling_mean(close, self.fast) - rolling_mean(close, self.slow)
        states = self._clip(np.nan_to_num(np.sign(diff), nan=0.0))
        return targets_from_states(bars[self.symbol].index, self.symbols, states,
                                   lambda i, s: {self.symbol: s})

    def on_bar(self, data: BarDataHandler) -> Weights | None:
        hist = data.history(self.symbol, n=self.slow)
        if len(hist) < self.slow:
            return None
        state = float(self._clip(np.sign(hist[-self.fast:].mean() - hist.mean())))
        if state == self._state:
            return None
        self._state = state
        return {self.symbol: state}
