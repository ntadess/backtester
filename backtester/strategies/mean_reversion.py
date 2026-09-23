from __future__ import annotations

import pandas as pd

from ..data import BarDataHandler, Bars
from .base import (Strategy, Weights, hysteresis_states, hysteresis_step,
                   rolling_zscore, targets_from_states, zscore_last)


class ZScoreMeanReversion(Strategy):
    """Fade moves: short when price is `entry` std devs above its rolling mean,
    long when below, flatten once |z| drops under `exit`."""

    def __init__(self, symbol: str, lookback: int = 20, entry: float = 2.0, exit: float = 0.5):
        if not 0 <= exit < entry:
            raise ValueError("need 0 <= exit < entry")
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.symbol, self.lookback, self.entry, self.exit = symbol, lookback, entry, exit
        self.symbols = (symbol,)
        self.reset()

    def reset(self) -> None:
        self._state = 0.0

    def generate_targets(self, bars: Bars) -> pd.DataFrame:
        close = bars[self.symbol]["close"].to_numpy()
        states = hysteresis_states(rolling_zscore(close, self.lookback), self.entry, self.exit)
        return targets_from_states(bars[self.symbol].index, self.symbols, states,
                                   lambda i, s: {self.symbol: s})

    def on_bar(self, data: BarDataHandler) -> Weights | None:
        hist = data.history(self.symbol, n=self.lookback)
        if len(hist) < self.lookback:
            return None
        state = hysteresis_step(zscore_last(hist), self.entry, self.exit, self._state)
        if state == self._state:
            return None
        self._state = state
        return {self.symbol: state}
