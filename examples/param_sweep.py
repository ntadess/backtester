"""Grid-search MA crossover parameters with the vectorized path, pick the
best on an in-sample window, then confirm out-of-sample with the
event-driven engine.

    python examples/param_sweep.py
"""
import itertools
import time

import pandas as pd

import backtester as bt

COSTS = bt.CostModel(slippage_bps=2.0, commission_bps=1.0)


def main() -> None:
    bars = bt.synthetic.trending(n=252 * 12, seed=7)
    index = bars["TREND"].index
    split = index[int(len(index) * 0.7)]
    print(f"in-sample: {index[0].date()} .. {split.date()}   "
          f"out-of-sample: {split.date()} .. {index[-1].date()}\n")

    fasts, slows = [5, 10, 20, 40, 60], [50, 100, 150, 200, 250]
    rows = []
    t0 = time.perf_counter()
    for fast, slow in itertools.product(fasts, slows):
        if fast >= slow:
            continue
        res = bt.run_vectorized(bars, bt.MovingAverageCrossover("TREND", fast, slow), costs=COSTS)
        rows.append({"fast": fast, "slow": slow,
                     "is_sharpe": res.slice(end=split).metrics()["sharpe"]})
    elapsed = time.perf_counter() - t0
    grid = pd.DataFrame(rows).pivot(index="fast", columns="slow", values="is_sharpe")
    print(f"In-sample Sharpe grid ({len(rows)} backtests in {elapsed:.2f}s):")
    print(grid.round(2).to_string(), "\n")

    best = max(rows, key=lambda r: r["is_sharpe"])
    strat = bt.MovingAverageCrossover("TREND", best["fast"], best["slow"])
    # Run over the full history so indicators are warmed up at the split,
    # but score only the out-of-sample window: parameters never saw it.
    full = bt.EventDrivenBacktest(bars, strat, costs=COSTS).run()
    ins, oos = full.slice(end=split), full.slice(start=split)
    print(f"Best in-sample: fast={best['fast']} slow={best['slow']}")
    print(bt.metrics_table({"in-sample": ins, "out-of-sample": oos}).to_string())
    print("\nExpect out-of-sample Sharpe below in-sample: that gap is the price "
          "of selecting the best of many trials.")


if __name__ == "__main__":
    main()
