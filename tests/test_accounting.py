"""Hand-checked cash, cost, and timing accounting."""
import numpy as np
import pytest

from backtester import CostModel, EventDrivenBacktest, run_vectorized
from backtester.metrics import max_drawdown
from conftest import ScheduledWeights, flat_bars

import pandas as pd


@pytest.mark.parametrize("engine", ["event", "vectorized"])
def test_round_trip_costs_and_timing(engine):
    # prices: decide at bar 0 close (100), fill at bar 1 open (100), exit decided bar 3
    bars = flat_bars([100, 100, 110, 120, 120, 120])
    strat = ScheduledWeights("A", {0: 1.0, 3: 0.0})
    costs = CostModel(slippage_bps=10, commission_bps=5)
    kw = dict(initial_capital=10_000, costs=costs)
    res = (EventDrivenBacktest(bars, strat, **kw).run() if engine == "event"
           else run_vectorized(bars, strat, **kw))

    buy_px = 100 * 1.001
    buy_comm = 100 * buy_px * 5e-4
    sell_px = 120 * 0.999
    sell_comm = 100 * sell_px * 5e-4

    assert list(res.positions["A"]) == [0, 100, 100, 100, 0, 0]  # entry bar 1, exit bar 4
    assert res.equity.iloc[0] == pytest.approx(10_000)
    assert res.equity.iloc[1] == pytest.approx(10_000 - 100 * buy_px - buy_comm + 100 * 100)
    final = 10_000 - 100 * buy_px - buy_comm + 100 * sell_px - sell_comm
    assert res.equity.iloc[-1] == pytest.approx(final)
    assert res.trades["commission"].sum() == pytest.approx(buy_comm + sell_comm)
    assert res.trades["slippage"].sum() == pytest.approx(100 * 0.1 + 100 * 0.12)


def test_short_position_pnl():
    bars = flat_bars([100, 100, 90, 80])
    res = EventDrivenBacktest(bars, ScheduledWeights("A", {0: -1.0}), initial_capital=10_000).run()
    assert res.positions["A"].iloc[-1] == -100
    assert res.equity.iloc[-1] == pytest.approx(12_000)


def test_signal_on_last_bar_never_fills():
    bars = flat_bars([100, 100, 100])
    res = EventDrivenBacktest(bars, ScheduledWeights("A", {2: 1.0}), initial_capital=10_000).run()
    assert res.trades.empty


def test_equity_vs_fixed_sizing():
    # 100 shares at 100, price doubles to 200 (equity 20k), then cut to 50% weight
    bars = flat_bars([100, 100, 200, 200, 200, 200])
    strat = ScheduledWeights("A", {0: 1.0, 3: 0.5})
    fixed = EventDrivenBacktest(bars, strat, initial_capital=10_000, sizing="fixed").run()
    comp = EventDrivenBacktest(bars, strat, initial_capital=10_000, sizing="equity").run()
    assert fixed.positions["A"].iloc[-1] == 25   # 0.5 * 10k / 200
    assert comp.positions["A"].iloc[-1] == 50    # 0.5 * 20k / 200


def test_max_drawdown():
    eq = pd.Series([100.0, 120, 90, 100, 130])
    mdd, bars = max_drawdown(eq)
    assert mdd == pytest.approx(-0.25)
    assert bars == 2
