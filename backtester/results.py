from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import compute_metrics, format_metrics

TRADE_COLUMNS = ["timestamp", "symbol", "quantity", "price", "commission", "slippage"]


@dataclass
class BacktestResult:
    equity: pd.Series          # mark-to-market at each bar's close
    positions: pd.DataFrame    # shares held at each bar's close
    trades: pd.DataFrame       # one row per fill, columns = TRADE_COLUMNS
    initial_capital: float
    periods_per_year: int = 252

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def drawdown(self) -> pd.Series:
        return self.equity / self.equity.cummax() - 1.0

    def metrics(self) -> dict[str, float]:
        return compute_metrics(self.equity, self.trades, self.positions, self.periods_per_year)

    def slice(self, start=None, end=None) -> "BacktestResult":
        """Restrict to a date range, e.g. to score an out-of-sample window."""
        eq = self.equity.loc[start:end]
        t = self.trades
        mask = (t["timestamp"] >= eq.index[0]) & (t["timestamp"] <= eq.index[-1])
        return BacktestResult(eq, self.positions.loc[start:end], t[mask].reset_index(drop=True),
                              float(eq.iloc[0]), self.periods_per_year)

    def summary(self) -> str:
        m = format_metrics(self.metrics())
        width = max(map(len, m))
        return "\n".join(f"{k:<{width}}  {v:>12}" for k, v in m.items())


def metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Side-by-side formatted metrics for several named results."""
    return pd.DataFrame({name: format_metrics(r.metrics()) for name, r in results.items()})
