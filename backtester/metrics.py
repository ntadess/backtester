"""Performance statistics computed from an equity curve and a trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(worst peak-to-trough loss as a negative fraction, longest underwater run in bars)."""
    dd = equity / equity.cummax() - 1.0
    underwater = (dd < 0).to_numpy()
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(dd.min()), longest


def compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                    positions: pd.DataFrame | None = None,
                    periods_per_year: int = 252) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    n = len(rets)
    years = n / periods_per_year if n else np.nan
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total = end / start - 1.0
    cagr = (end / start) ** (1.0 / years) - 1.0 if n and end > 0 else np.nan

    sd = rets.std(ddof=1) if n > 1 else np.nan
    ann = np.sqrt(periods_per_year)
    sharpe = rets.mean() / sd * ann if sd and sd > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(rets.to_numpy(), 0.0) ** 2)) if n else np.nan
    sortino = rets.mean() / downside * ann if downside and downside > 0 else np.nan
    mdd, mdd_len = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    notional = float((trades["quantity"].abs() * trades["price"]).sum()) if len(trades) else 0.0
    costs = float((trades["commission"] + trades["slippage"]).sum()) if len(trades) else 0.0
    turnover = notional / equity.mean() / years if years and years > 0 else np.nan
    exposure = (float((positions.abs().sum(axis=1) > 0).mean())
                if positions is not None and len(positions) else np.nan)

    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": sd * ann if n > 1 else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_dd_bars": float(mdd_len),
        "calmar": calmar,
        "ann_turnover": turnover,
        "exposure": exposure,
        "n_fills": float(len(trades)),
        "total_costs": costs,
    }


_PCT = {"total_return", "cagr", "ann_vol", "max_drawdown", "exposure"}


def format_metrics(metrics: dict[str, float]) -> dict[str, str]:
    out = {}
    for k, v in metrics.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[k] = "n/a"
        elif k in _PCT:
            out[k] = f"{v:.2%}"
        elif k in {"n_fills", "max_dd_bars"}:
            out[k] = f"{int(v):,}"
        elif k == "total_costs":
            out[k] = f"${v:,.0f}"
        else:
            out[k] = f"{v:.2f}"
    return out
