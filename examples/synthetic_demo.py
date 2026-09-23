"""Run every strategy on synthetic data through both engines.

    python examples/synthetic_demo.py

Prints metrics, the max equity difference between engines (should be ~1e-9),
the speedup of the vectorized path, and saves equity/drawdown plots to output/.
"""
import time
from pathlib import Path

import backtester as bt

COSTS = bt.CostModel(slippage_bps=2.0, commission_bps=1.0)

CASES = {
    "MA crossover (trending)": (bt.synthetic.trending(seed=1),
                                bt.MovingAverageCrossover("TREND", fast=20, slow=100)),
    "Z-score MR (OU)": (bt.synthetic.mean_reverting(seed=2),
                        bt.ZScoreMeanReversion("OU", lookback=20, entry=2.0, exit=0.5)),
    "Pairs (cointegrated)": (bt.synthetic.cointegrated_pair(seed=3),
                             bt.PairsTrading("Y", "X", lookback=250, z_window=60)),
}


def main() -> None:
    out = Path("output")
    out.mkdir(exist_ok=True)
    results = {}
    for name, (bars, strategy) in CASES.items():
        t0 = time.perf_counter()
        ev = bt.EventDrivenBacktest(bars, strategy, costs=COSTS).run()
        t1 = time.perf_counter()
        vec = bt.run_vectorized(bars, strategy, costs=COSTS)
        t2 = time.perf_counter()
        gap = (ev.equity - vec.equity).abs().max()
        print(f"{name:<26} parity gap ${gap:.2e}   event {t1 - t0:6.3f}s   "
              f"vectorized {t2 - t1:6.3f}s   ({(t1 - t0) / (t2 - t1):.0f}x)")
        results[name] = ev
        try:
            from backtester.plotting import plot_result
            slug = name.split(" (")[0].lower().replace(" ", "_").replace("-", "")
            plot_result(ev, name, out / f"{slug}.png")
        except ImportError:
            pass

    print()
    print(bt.metrics_table(results).to_string())


if __name__ == "__main__":
    main()
