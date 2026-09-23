"""Optional matplotlib plots (pip install matplotlib)."""
from __future__ import annotations

from pathlib import Path

from .results import BacktestResult


def plot_result(result: BacktestResult, title: str = "", path: str | Path | None = None):
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(result.equity.index, result.equity.to_numpy(), lw=1.2)
    ax1.set_title(title)
    ax1.set_ylabel("Equity")
    ax1.grid(alpha=0.3)
    dd = result.drawdown
    ax2.fill_between(dd.index, dd.to_numpy() * 100, 0, alpha=0.4, color="tab:red")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig
