from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import BarDataHandler, Bars
from .base import (Strategy, Weights, hysteresis_states, hysteresis_step, ols_last,
                   rolling_ols, rolling_zscore, targets_from_states, zscore_last)


class PairsTrading(Strategy):
    """Rolling-OLS statistical arbitrage on two instruments.

    Each bar: regress y on x over the trailing `lookback` bars (only past
    data, so no lookahead in the hedge ratio), form
        spread = y - (alpha + beta * x),
    and z-score the spread over the trailing `z_window` spreads.

    Long spread (+1) = buy y, sell beta * x shares; short spread is the reverse.
    Share counts are fixed at entry, i.e. the hedge ratio is not rebalanced
    while a position is open. `gross` sets y-leg notional to gross / 2.
    """

    def __init__(self, y: str, x: str, lookback: int = 60, z_window: int | None = None,
                 entry: float = 2.0, exit: float = 0.5, gross: float = 1.0):
        if not 0 <= exit < entry:
            raise ValueError("need 0 <= exit < entry")
        self.y, self.x = y, x
        self.lookback = lookback
        self.z_window = z_window or lookback
        self.entry, self.exit, self.gross = entry, exit, gross
        self.symbols = (y, x)
        self.reset()

    def reset(self) -> None:
        self._state = 0.0
        self._spreads: list[float] = []

    def _weights(self, state: float, beta: float, px_y: float, px_x: float) -> Weights:
        if state == 0.0:
            return {self.y: 0.0, self.x: 0.0}
        w_y = state * self.gross / 2.0
        # fixed-notional sizing: shares = w * capital / price, so this gives
        # shares_x = -beta * shares_y
        w_x = -w_y * beta * px_x / px_y
        return {self.y: w_y, self.x: w_x}

    def generate_targets(self, bars: Bars) -> pd.DataFrame:
        y = bars[self.y]["close"].to_numpy()
        x = bars[self.x]["close"].to_numpy()
        beta, alpha = rolling_ols(y, x, self.lookback)
        spread = y - (alpha + beta * x)
        states = hysteresis_states(rolling_zscore(spread, self.z_window), self.entry, self.exit)
        return targets_from_states(bars[self.y].index, self.symbols, states,
                                   lambda i, s: self._weights(s, beta[i], y[i], x[i]))

    def on_bar(self, data: BarDataHandler) -> Weights | None:
        yh = data.history(self.y, n=self.lookback)
        xh = data.history(self.x, n=self.lookback)
        if len(yh) < self.lookback:
            return None
        beta, alpha = ols_last(yh, xh)
        spread = yh[-1] - (alpha + beta * xh[-1])
        self._spreads.append(spread)
        if len(self._spreads) < self.z_window:
            return None
        z = zscore_last(np.asarray(self._spreads[-self.z_window:]))
        state = hysteresis_step(z, self.entry, self.exit, self._state)
        if state == self._state:
            return None
        self._state = state
        return self._weights(state, beta, float(yh[-1]), float(xh[-1]))
