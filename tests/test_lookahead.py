"""Changing the future must not change the past."""
import numpy as np
import pytest

from backtester import (BarDataHandler, EventDrivenBacktest, MovingAverageCrossover,
                        PairsTrading, ZScoreMeanReversion, run_vectorized, synthetic)


def perturb_after(bars, k, seed=99):
    rng = np.random.default_rng(seed)
    out = {}
    for sym, df in bars.items():
        df = df.copy()
        shock = np.exp(rng.normal(0, 0.05, len(df) - k))
        for col in ("open", "high", "low", "close"):
            df.iloc[k:, df.columns.get_loc(col)] *= shock
        out[sym] = df
    return out


CASES = [
    (lambda: synthetic.trending(n=1200, seed=4), lambda: MovingAverageCrossover("TREND", 10, 50)),
    (lambda: synthetic.mean_reverting(n=1200, seed=4), lambda: ZScoreMeanReversion("OU", 20)),
    (lambda: synthetic.cointegrated_pair(n=1200, seed=4), lambda: PairsTrading("Y", "X", 60, 30)),
]


@pytest.mark.parametrize("runner", ["event", "vectorized"])
@pytest.mark.parametrize("make_bars,make_strat", CASES, ids=["ma", "zscore", "pairs"])
def test_future_changes_do_not_affect_past(make_bars, make_strat, runner):
    k = 800
    bars = make_bars()
    run = (lambda b: EventDrivenBacktest(b, make_strat()).run()) if runner == "event" \
        else (lambda b: run_vectorized(b, make_strat()))
    base, shocked = run(bars), run(perturb_after(bars, k))
    # equity through bar k-1 is fully determined by bars 0..k-1
    np.testing.assert_array_equal(base.equity.iloc[:k].to_numpy(), shocked.equity.iloc[:k].to_numpy())
    assert not np.array_equal(base.equity.to_numpy(), shocked.equity.to_numpy())


def test_handler_exposes_only_past():
    bars = synthetic.gbm(n=50, seed=0)
    data = BarDataHandler(bars)
    closes = bars["GBM"]["close"].to_numpy()
    for i in range(50):
        data.advance()
        hist = data.history("GBM")
        assert len(hist) == i + 1
        assert hist[-1] == closes[i]
        with pytest.raises(ValueError):
            hist[0] = 0.0  # read-only
    assert data.advance() is None
