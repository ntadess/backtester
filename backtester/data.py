"""Market data: alignment, a lookahead-proof bar handler, and loaders."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

Bars = dict[str, pd.DataFrame]
REQUIRED_COLUMNS = ("open", "high", "low", "close")


def align_bars(bars: Mapping[str, pd.DataFrame]) -> Bars:
    """Normalise column names, sort, drop NaN rows and intersect indexes.

    Every engine calls this, so both the vectorized and event-driven paths
    see exactly the same timestamps.
    """
    if not bars:
        raise ValueError("no bars provided")
    cleaned: Bars = {}
    for sym, df in bars.items():
        df = df.rename(columns=lambda c: str(c).lower())
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{sym}: missing columns {missing}")
        cols = [*REQUIRED_COLUMNS] + (["volume"] if "volume" in df.columns else [])
        df = df[cols].astype(float).sort_index()
        if df.index.has_duplicates:
            raise ValueError(f"{sym}: duplicate timestamps in index")
        cleaned[sym] = df.dropna(subset=list(REQUIRED_COLUMNS))

    common: pd.Index | None = None
    for df in cleaned.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) == 0:
        raise ValueError("symbols share no common timestamps")
    return {sym: df.loc[common] for sym, df in cleaned.items()}


class BarDataHandler:
    """Replays aligned bars one timestamp at a time.

    Strategies only ever get slices ending at the current bar, and the
    underlying arrays are read-only, so lookahead is structurally impossible
    rather than something each strategy has to remember to avoid.
    """

    def __init__(self, bars: Mapping[str, pd.DataFrame]):
        aligned = align_bars(bars)
        self.symbols: tuple[str, ...] = tuple(aligned)
        self.index: pd.Index = next(iter(aligned.values())).index
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        for sym, df in aligned.items():
            cols = {}
            for col in df.columns:
                arr = df[col].to_numpy(dtype=float, copy=True)
                arr.flags.writeable = False
                cols[col] = arr
            self._arrays[sym] = cols
        self._i = -1

    def __len__(self) -> int:
        return len(self.index)

    def reset(self) -> None:
        self._i = -1

    def advance(self) -> pd.Timestamp | None:
        """Move to the next bar; returns its timestamp, or None when done."""
        if self._i + 1 >= len(self.index):
            return None
        self._i += 1
        return self.index[self._i]

    @property
    def current_time(self) -> pd.Timestamp:
        if self._i < 0:
            raise RuntimeError("advance() has not been called yet")
        return self.index[self._i]

    @property
    def bars_seen(self) -> int:
        return self._i + 1

    def price(self, symbol: str, field: str = "close") -> float:
        """Value of `field` on the current bar."""
        if self._i < 0:
            raise RuntimeError("advance() has not been called yet")
        return float(self._arrays[symbol][field][self._i])

    def history(self, symbol: str, field: str = "close", n: int | None = None) -> np.ndarray:
        """Read-only view of `field` up to and including the current bar."""
        end = self._i + 1
        start = 0 if n is None else max(0, end - n)
        return self._arrays[symbol][field][start:end]


def load_csv(paths: Mapping[str, str | Path], **read_csv_kwargs) -> Bars:
    """Load one CSV per symbol. The first column is parsed as the datetime index."""
    kwargs = {"index_col": 0, "parse_dates": True, **read_csv_kwargs}
    return align_bars({sym: pd.read_csv(p, **kwargs) for sym, p in paths.items()})


def load_yfinance(
    symbols: Iterable[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> Bars:
    """Download OHLCV bars from Yahoo Finance (requires `pip install yfinance`).

    Caveats: free data has survivorship bias (delisted tickers are missing)
    and intraday history is short (about 730 days for 1h, 7 days for 1m).
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_yfinance needs yfinance: pip install yfinance") from exc

    out: Bars = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, interval=interval,
                         auto_adjust=auto_adjust, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise ValueError(f"no data returned for {sym}")
        out[sym] = df
    return align_bars(out)
