"""Real-market examples with Yahoo Finance data.

    pip install yfinance
    python examples/yfinance_demo.py --start 2010-01-01

These are illustrations, not recommendations: free Yahoo data has
survivorship bias, and KO/PEP is a textbook pair whose relationship is not
guaranteed to hold.
"""
import argparse

import numpy as np
import pandas as pd

import backtester as bt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--plot", action="store_true", help="save PNGs to output/")
    args = ap.parse_args()

    costs = bt.CostModel(slippage_bps=1.0, commission_per_share=0.005)
    cases = {
        "SPY 50/200 long-only": (["SPY"], bt.MovingAverageCrossover("SPY", 50, 200, long_only=True)),
        "QQQ z-score MR": (["QQQ"], bt.ZScoreMeanReversion("QQQ", lookback=20)),
        "KO/PEP pairs": (["KO", "PEP"], bt.PairsTrading("KO", "PEP", lookback=250, z_window=60)),
    }

    results = {}
    for name, (tickers, strategy) in cases.items():
        bars = bt.load_yfinance(tickers, start=args.start, end=args.end)
        ev = bt.EventDrivenBacktest(bars, strategy, costs=costs).run()
        vec = bt.run_vectorized(bars, strategy, costs=costs)
        print(f"{name:<22} parity gap ${(ev.equity - vec.equity).abs().max():.2e}")
        results[name] = ev
        if args.plot:
            from pathlib import Path
            from backtester.plotting import plot_result
            Path("output").mkdir(exist_ok=True)
            slug = name.split()[0].lower().replace("/", "_")
            plot_result(ev, name, f"output/{slug}_{type(strategy).__name__}.png")

    spy = bt.load_yfinance(["SPY"], start=args.start, end=args.end)
    results["SPY buy & hold"] = bt.EventDrivenBacktest(spy, BuyAndHold("SPY"), costs=costs).run()

    print()
    print(bt.metrics_table(results).to_string())


class BuyAndHold(bt.Strategy):
    """Benchmark: go 100% long on the first bar and never trade again."""

    def __init__(self, symbol: str):
        self.symbols = (symbol,)

    def generate_targets(self, bars):
        sym = self.symbols[0]
        t = pd.DataFrame(np.nan, index=bars[sym].index, columns=[sym])
        t.iloc[0, 0] = 1.0
        return t

    def on_bar(self, data):
        return {self.symbols[0]: 1.0} if data.bars_seen == 1 else None


if __name__ == "__main__":
    main()
