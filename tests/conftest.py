import numpy as np
import pandas as pd
import pytest

from backtester import synthetic
from backtester.strategies.base import Strategy


class ScheduledWeights(Strategy):
    """Test helper: emits pre-set weights on given bar numbers."""

    def __init__(self, symbol: str, schedule: dict[int, float]):
        self.symbol, self.schedule = symbol, schedule
        self.symbols = (symbol,)

    def generate_targets(self, bars):
        idx = bars[self.symbol].index
        t = pd.DataFrame(np.nan, index=idx, columns=[self.symbol])
        for i, w in self.schedule.items():
            t.iloc[i, 0] = w
        return t

    def on_bar(self, data):
        w = self.schedule.get(data.bars_seen - 1)
        return None if w is None else {self.symbol: w}


def flat_bars(prices, symbol="A"):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    p = np.asarray(prices, float)
    return {symbol: pd.DataFrame({"open": p, "high": p, "low": p, "close": p}, index=idx)}


@pytest.fixture
def trend_bars():
    return synthetic.trending(n=1500, seed=1)


@pytest.fixture
def ou_bars():
    return synthetic.mean_reverting(n=1500, seed=2)


@pytest.fixture
def pair_bars():
    return synthetic.cointegrated_pair(n=1500, seed=3)
