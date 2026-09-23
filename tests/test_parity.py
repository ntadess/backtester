"""The vectorized fast path must reproduce the event-driven engine exactly."""
import numpy as np
import pandas as pd
import pytest

from backtester import (CostModel, EventDrivenBacktest, MovingAverageCrossover, PairsTrading,
                        ZScoreMeanReversion, run_vectorized, synthetic)
from backtester.strategies.base import Strategy

COSTS = CostModel(slippage_bps=2.0, commission_bps=1.0, commission_per_share=0.005)


def assert_same(a, b):
    np.testing.assert_allclose(a.equity.to_numpy(), b.equity.to_numpy(), rtol=1e-10)
    pd.testing.assert_frame_equal(a.positions, b.positions, check_names=False)
    key = ["timestamp", "symbol"]
    ta = a.trades.sort_values(key).reset_index(drop=True)
    tb = b.trades.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(ta, tb, check_exact=False, rtol=1e-12)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("make", [
    lambda s: (synthetic.trending(n=1500, seed=s), MovingAverageCrossover("TREND", 10, 50)),
    lambda s: (synthetic.trending(n=1500, seed=s), MovingAverageCrossover("TREND", 20, 100, long_only=True)),
    lambda s: (synthetic.mean_reverting(n=1500, seed=s), ZScoreMeanReversion("OU", 20, 2.0, 0.5)),
    lambda s: (synthetic.cointegrated_pair(n=1500, seed=s), PairsTrading("Y", "X", 60, 30)),
], ids=["ma", "ma_long_only", "zscore", "pairs"])
def test_engines_agree(make, seed):
    bars, strat = make(seed)
    ev = EventDrivenBacktest(bars, strat, costs=COSTS).run()
    vec = run_vectorized(bars, strat, costs=COSTS)
    assert len(ev.trades) > 0
    assert_same(ev, vec)


def test_fractional_shares_agree(pair_bars):
    strat = PairsTrading("Y", "X", 60, 30)
    ev = EventDrivenBacktest(pair_bars, strat, costs=COSTS, integer_shares=False).run()
    vec = run_vectorized(pair_bars, strat, costs=COSTS, integer_shares=False)
    assert_same(ev, vec)


class PeekingStrategy(Strategy):
    """A buggy vectorized strategy that uses tomorrow's return.

    Its event-driven twin cannot see the future, so the parity check fails.
    This is exactly the class of bug the cross-check exists to catch.
    """

    symbols = ("TREND",)

    def generate_targets(self, bars):
        close = bars["TREND"]["close"]
        state = np.sign(close.shift(-1) - close).fillna(0.0)  # lookahead!
        t = state.where(state != state.shift(1).fillna(0.0))
        return t.to_frame("TREND")

    def on_bar(self, data):
        return None  # honest version has nothing to act on


def test_parity_catches_lookahead(trend_bars):
    ev = EventDrivenBacktest(trend_bars, PeekingStrategy()).run()
    vec = run_vectorized(trend_bars, PeekingStrategy())
    assert vec.equity.iloc[-1] > ev.equity.iloc[-1] * 1.5  # "too good to be true"
    with pytest.raises(AssertionError):
        assert_same(ev, vec)
