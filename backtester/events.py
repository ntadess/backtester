"""Event types that flow through the event-driven engine.

Lifecycle of one bar:
    MarketEvent  -> execution fills pending orders at this bar's open (FillEvent)
                 -> strategy sees bars up to and including this one (SignalEvent)
    SignalEvent  -> portfolio turns target weights into share deltas (OrderEvent)
    OrderEvent   -> execution queues the order; it fills at the NEXT bar's open
    FillEvent    -> portfolio updates cash and positions
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketEvent:
    timestamp: pd.Timestamp


@dataclass(frozen=True, slots=True)
class SignalEvent:
    timestamp: pd.Timestamp
    target_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class OrderEvent:
    timestamp: pd.Timestamp  # decision time
    symbol: str
    quantity: float  # signed: + buy, - sell


@dataclass(frozen=True, slots=True)
class FillEvent:
    timestamp: pd.Timestamp  # fill time
    symbol: str
    quantity: float
    price: float
    commission: float
    slippage: float  # dollar cost of slippage vs. reference price


Event = MarketEvent | SignalEvent | OrderEvent | FillEvent
