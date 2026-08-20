"""Rebuild data/indices.csv: daily closes for the four Gidea-Katz indices
(S&P 500, Dow, NASDAQ Composite, Russell 2000) from Yahoo Finance, inner-joined
on common trading days from 1997 onward.

Usage:
    python scripts/fetch_indices.py
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {"^GSPC": "spx", "^DJI": "dji", "^IXIC": "ixic", "^RUT": "rut"}
START = 852076800  # 1997-01-01


def fetch(symbol: str) -> pd.Series:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(symbol)}?period1={START}"
           f"&period2={int(time.time())}&interval=1d")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    with urllib.request.urlopen(req) as resp:
        r = json.load(resp)["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s", utc=True)
    dates = pd.to_datetime(ts.tz_convert("America/New_York").date)
    s = pd.Series(r["indicators"]["quote"][0]["close"], index=dates,
                  name=SYMBOLS[symbol]).dropna()
    return s[~s.index.duplicated(keep="last")]


def main():
    frames = []
    for sym, name in SYMBOLS.items():
        s = fetch(sym)
        print(f"{name}: {len(s)} rows, {s.index[0].date()} .. {s.index[-1].date()}")
        frames.append(s)
        time.sleep(1)
    df = pd.concat(frames, axis=1, join="inner").sort_index()
    df.index.name = "date"
    out = ROOT / "data/indices.csv"
    df.round(4).to_csv(out, date_format="%Y-%m-%d")
    print(f"merged: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
