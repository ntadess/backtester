"""Strategies should find edges we planted in synthetic data, and not
find much in data with no edge."""
import numpy as np

from backtester import (MovingAverageCrossover, PairsTrading, ZScoreMeanReversion,
                        run_vectorized, synthetic)


def mean_sharpe(make_bars, strat, seeds=range(5)):
    return np.mean([run_vectorized(make_bars(s), strat).metrics()["sharpe"] for s in seeds])


def test_mean_reversion_profits_on_ou():
    assert mean_sharpe(lambda s: synthetic.mean_reverting(seed=s), ZScoreMeanReversion("OU")) > 1.0


def test_pairs_profits_on_cointegrated_pair():
    assert mean_sharpe(lambda s: synthetic.cointegrated_pair(seed=s),
                       PairsTrading("Y", "X", lookback=250, z_window=60)) > 0.8


def test_trend_following_beats_random_walk():
    strat = MovingAverageCrossover("S", 20, 100)
    trend = mean_sharpe(lambda s: synthetic.trending(seed=s, symbol="S"), strat)
    walk = mean_sharpe(lambda s: synthetic.gbm(seed=s, symbol="S", mu=0.0), strat)
    assert trend > 0.5
    assert trend > walk + 0.4


def test_synthetic_ohlc_is_consistent():
    for bars in (synthetic.gbm(), synthetic.trending(), synthetic.mean_reverting(),
                 synthetic.cointegrated_pair()):
        for df in bars.values():
            assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
            assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
            assert (df["low"] > 0).all()
