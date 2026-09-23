"""Transaction cost model shared by both engines (scalars or arrays)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Slippage moves the fill price against you; commission is charged on top.

    slippage_bps:         fill = ref * (1 + sign(qty) * bps / 1e4)
    commission_bps:       charged on traded notional
    commission_per_share: flat fee per share (e.g. 0.005 for US equities)
    """

    slippage_bps: float = 0.0
    commission_bps: float = 0.0
    commission_per_share: float = 0.0

    def fill_price(self, ref_price, qty):
        return ref_price * (1.0 + np.sign(qty) * self.slippage_bps * 1e-4)

    def commission(self, qty, price):
        return np.abs(qty) * (price * self.commission_bps * 1e-4 + self.commission_per_share)

    @staticmethod
    def slippage_cost(qty, ref_price, price):
        return np.abs(qty) * np.abs(price - ref_price)
