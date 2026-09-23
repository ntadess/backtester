"""Vectorized fast path: whole-history array math, for parameter sweeps.

Implements the same conventions as the event-driven engine:
  * targets decided on bar t's close are filled at bar t+1's open
  * fixed-notional sizing, shares = round(weight * capital / close_t)
  * identical cost model
so the two engines should agree to floating-point precision.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .costs import CostModel
from .data import align_bars
from .results import TRADE_COLUMNS, BacktestResult
from .strategies.base import Strategy


def run_vectorized(bars: Mapping[str, pd.DataFrame], strategy: Strategy,
                   initial_capital: float = 1_000_000.0, costs: CostModel | None = None,
                   integer_shares: bool = True, periods_per_year: int = 252) -> BacktestResult:
    costs = costs or CostModel()
    aligned = align_bars(bars)
    syms = list(strategy.symbols)
    missing = set(syms) - set(aligned)
    if missing:
        raise KeyError(f"no data for {missing}")
    index = aligned[syms[0]].index

    targets = strategy.generate_targets(aligned)
    if not targets.index.equals(index):
        raise ValueError("strategy targets must be indexed like the bars")
    targets = targets.reindex(columns=syms)

    close = pd.DataFrame({s: aligned[s]["close"] for s in syms})
    opens = pd.DataFrame({s: aligned[s]["open"] for s in syms})

    shares = targets * initial_capital / close        # only on instruction rows
    if integer_shares:
        shares = np.round(shares)
    # hold between instructions, then shift: decided at t, held from t+1's open
    held = shares.ffill().fillna(0.0).shift(1).fillna(0.0)
    qty = held.diff()
    qty.iloc[0] = held.iloc[0]

    fill = costs.fill_price(opens, qty)
    commission = costs.commission(qty, fill)
    slippage = costs.slippage_cost(qty, opens, fill)

    cash = initial_capital - (qty * fill + commission).sum(axis=1).cumsum()
    equity = (cash + (held * close).sum(axis=1)).rename("equity")

    q = qty.to_numpy()
    ti, si = np.nonzero(q)
    trades = pd.DataFrame({
        "timestamp": index[ti],
        "symbol": np.asarray(syms, dtype=object)[si],
        "quantity": q[ti, si],
        "price": fill.to_numpy()[ti, si],
        "commission": commission.to_numpy()[ti, si],
        "slippage": slippage.to_numpy()[ti, si],
    }, columns=TRADE_COLUMNS)

    return BacktestResult(equity, held, trades, float(initial_capital), periods_per_year)
