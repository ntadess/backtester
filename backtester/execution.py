from __future__ import annotations

from .costs import CostModel
from .data import BarDataHandler
from .events import FillEvent, MarketEvent, OrderEvent


class SimulatedExecution:
    """Fills orders at the next bar's open, adjusted by the cost model.

    Orders created while processing bar t sit in `pending` and are filled
    when bar t+1 arrives, so a strategy can never trade at a price it used
    to make the decision. Orders still pending after the last bar are dropped.
    """

    def __init__(self, data: BarDataHandler, costs: CostModel):
        self.data = data
        self.costs = costs
        self.pending: list[OrderEvent] = []

    def on_order(self, order: OrderEvent) -> None:
        self.pending.append(order)

    def on_market(self, event: MarketEvent) -> list[FillEvent]:
        fills = []
        for order in self.pending:
            ref = self.data.price(order.symbol, "open")
            price = float(self.costs.fill_price(ref, order.quantity))
            fills.append(FillEvent(
                timestamp=event.timestamp,
                symbol=order.symbol,
                quantity=order.quantity,
                price=price,
                commission=float(self.costs.commission(order.quantity, price)),
                slippage=float(self.costs.slippage_cost(order.quantity, ref, price)),
            ))
        self.pending.clear()
        return fills
