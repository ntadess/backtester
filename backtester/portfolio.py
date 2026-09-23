from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BarDataHandler
from .events import FillEvent, OrderEvent, SignalEvent
from .results import TRADE_COLUMNS, BacktestResult


class Portfolio:
    """Tracks cash and positions; converts target weights into orders.

    sizing="fixed":  target shares = weight * initial_capital / close
    sizing="equity": target shares = weight * current equity / close
                     (compounding; event-driven engine only)
    """

    def __init__(self, data: BarDataHandler, symbols: tuple[str, ...],
                 initial_capital: float, sizing: str = "fixed", integer_shares: bool = True):
        if sizing not in ("fixed", "equity"):
            raise ValueError("sizing must be 'fixed' or 'equity'")
        self.data = data
        self.symbols = symbols
        self.initial_capital = float(initial_capital)
        self.sizing = sizing
        self.integer_shares = integer_shares
        self.cash = float(initial_capital)
        self.positions: dict[str, float] = {s: 0.0 for s in symbols}
        self._equity: list[float] = []
        self._pos_rows: list[list[float]] = []
        self._times: list[pd.Timestamp] = []
        self._fills: list[FillEvent] = []

    def equity(self) -> float:
        return self.cash + sum(q * self.data.price(s) for s, q in self.positions.items())

    def on_signal(self, event: SignalEvent) -> list[OrderEvent]:
        unknown = set(event.target_weights) - set(self.symbols)
        if unknown:
            raise KeyError(f"signal for symbols the strategy did not declare: {unknown}")
        base = self.initial_capital if self.sizing == "fixed" else self.equity()
        orders = []
        for sym, w in event.target_weights.items():
            target = w * base / self.data.price(sym)
            if self.integer_shares:
                target = float(np.round(target))
            delta = target - self.positions[sym]
            if delta != 0:
                orders.append(OrderEvent(event.timestamp, sym, delta))
        return orders

    def on_fill(self, fill: FillEvent) -> None:
        self.positions[fill.symbol] += fill.quantity
        self.cash -= fill.quantity * fill.price + fill.commission
        self._fills.append(fill)

    def mark(self, timestamp: pd.Timestamp) -> None:
        """Snapshot equity and positions at the bar's close."""
        self._times.append(timestamp)
        self._equity.append(self.equity())
        self._pos_rows.append([self.positions[s] for s in self.symbols])

    def to_result(self, periods_per_year: int) -> BacktestResult:
        idx = self.data.index[:len(self._times)]
        trades = pd.DataFrame(
            [(f.timestamp, f.symbol, f.quantity, f.price, f.commission, f.slippage)
             for f in self._fills],
            columns=TRADE_COLUMNS,
        )
        return BacktestResult(
            equity=pd.Series(self._equity, index=idx, name="equity"),
            positions=pd.DataFrame(self._pos_rows, index=idx, columns=list(self.symbols)),
            trades=trades,
            initial_capital=self.initial_capital,
            periods_per_year=periods_per_year,
        )
