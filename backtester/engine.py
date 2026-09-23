"""Event-driven backtest engine: the source of truth."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import pandas as pd

from .costs import CostModel
from .data import BarDataHandler
from .events import Event, FillEvent, MarketEvent, OrderEvent, SignalEvent
from .execution import SimulatedExecution
from .portfolio import Portfolio
from .results import BacktestResult
from .strategies.base import Strategy


class EventDrivenBacktest:
    def __init__(self, bars: Mapping[str, pd.DataFrame], strategy: Strategy,
                 initial_capital: float = 1_000_000.0, costs: CostModel | None = None,
                 sizing: str = "fixed", integer_shares: bool = True,
                 periods_per_year: int = 252):
        self.bars = bars
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.costs = costs or CostModel()
        self.sizing = sizing
        self.integer_shares = integer_shares
        self.periods_per_year = periods_per_year

    def run(self) -> BacktestResult:
        data = BarDataHandler(self.bars)
        missing = set(self.strategy.symbols) - set(data.symbols)
        if missing:
            raise KeyError(f"no data for {missing}")
        self.strategy.reset()
        portfolio = Portfolio(data, self.strategy.symbols, self.initial_capital,
                              self.sizing, self.integer_shares)
        execution = SimulatedExecution(data, self.costs)
        queue: deque[Event] = deque()

        while (ts := data.advance()) is not None:
            queue.append(MarketEvent(ts))
            while queue:
                event = queue.popleft()
                if isinstance(event, MarketEvent):
                    # Fills are queued before the signal, so the portfolio
                    # sizes new orders against post-fill positions.
                    queue.extend(execution.on_market(event))
                    weights = self.strategy.on_bar(data)
                    if weights is not None:
                        queue.append(SignalEvent(ts, weights))
                elif isinstance(event, FillEvent):
                    portfolio.on_fill(event)
                elif isinstance(event, SignalEvent):
                    queue.extend(portfolio.on_signal(event))
                elif isinstance(event, OrderEvent):
                    execution.on_order(event)
            portfolio.mark(ts)

        return portfolio.to_result(self.periods_per_year)
