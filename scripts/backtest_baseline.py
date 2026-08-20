"""Backtest: TDA spike flags vs. a realized-volatility baseline.

Answers the obvious skeptic's question — does the topological signal add
anything over a plain volatility filter? Both detectors are evaluated causally
(no lookahead) on the same dates, against objectively defined crash events.

Definitions (all in trading days):
  Crash event   First day the drawdown from the trailing peak crosses below
                -15%. One event per drawdown episode; the episode ends when the
                price fully recovers its prior peak.
  Recall        Fraction of events with >= 1 flag in the 60 days before onset.
  Lead time     Onset minus the earliest flag in that 60-day pre-window.
  Precision     Fraction of flags landing within [onset-60, onset+60] of any
                event (flags during the crash itself count as useful).

Detectors:
  TDA           Spike column from the C++ engine (causal Welford z > 2 on the
                L2 delta between consecutive Betti-0 curves).
  Vol           20-day realized volatility of log returns, causal expanding
                z-score, z > 2.
  Vol (matched) Same vol statistic, but flagging the top-N days by z-score
                where N = number of TDA flags, for an equal-budget comparison.

Usage:
    python scripts/backtest_baseline.py [--binary cpp/build/qtda_stream]
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRE_WINDOW = 60      # trading days before onset that count as early warning
POST_WINDOW = 60     # trading days after onset still counted useful (precision)
DRAWDOWN = -0.15     # event threshold on drawdown from trailing peak
VOL_WINDOW = 20      # realized-vol lookback
Z = 2.0
MIN_HIST = 10


def crash_episodes(prices: np.ndarray) -> list[dict]:
    """Drawdown episodes crossing DRAWDOWN from the trailing peak. One event
    per episode (ends on full recovery of the prior peak, or at end of data).
    Returns onset index, peak index, and trough index (lowest close before
    recovery)."""
    peak_i = 0
    in_episode = False
    episodes = []
    for i, p in enumerate(prices):
        if p >= prices[peak_i]:
            peak_i = i
            in_episode = False
        elif not in_episode and p / prices[peak_i] - 1.0 < DRAWDOWN:
            in_episode = True
            episodes.append({"onset": i, "peak": peak_i, "trough": i})
        elif in_episode and p < prices[episodes[-1]["trough"]]:
            episodes[-1]["trough"] = i
    return episodes


def rolling_z(x: np.ndarray, win: int = 252) -> np.ndarray:
    """Causal z-score against a rolling window: sensitive to local buildup,
    where an expanding baseline grows desensitized after its first crisis."""
    s = pd.Series(x)
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return ((s - mu) / (sd + 1e-12)).to_numpy()


def causal_z(x: np.ndarray, min_hist: int = MIN_HIST) -> np.ndarray:
    """Expanding-window z-score using only data up to and including each day."""
    z = np.full(len(x), np.nan)
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        hist = x[: i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_hist:
            continue
        z[i] = (x[i] - hist.mean()) / (hist.std() + 1e-9)
    return z


def evaluate(flag_idx: np.ndarray, episodes: list[dict], prices: np.ndarray) -> dict:
    n_days = len(prices)
    flag_set = np.zeros(n_days, dtype=bool)
    flag_set[flag_idx] = True
    detected, leads, loss_ahead = 0, [], []
    for ep in episodes:
        ev = ep["onset"]
        pre = flag_set[max(0, ev - PRE_WINDOW): ev]
        if pre.any():
            detected += 1
            first = np.flatnonzero(pre)[0]
            leads.append(ev - (max(0, ev - PRE_WINDOW) + first))
        # In-episode value: how much of the eventual peak-to-trough loss was
        # still ahead when the first flag fired (between pre-window and trough)?
        span = np.flatnonzero(flag_set[max(0, ev - PRE_WINDOW): ep["trough"] + 1])
        if span.size:
            f = max(0, ev - PRE_WINDOW) + span[0]
            total = prices[ep["peak"]] - prices[ep["trough"]]
            loss_ahead.append((prices[f] - prices[ep["trough"]]) / total)
    useful = np.zeros(n_days, dtype=bool)
    for ep in episodes:
        ev = ep["onset"]
        useful[max(0, ev - PRE_WINDOW): min(n_days, ev + POST_WINDOW + 1)] = True
    tp = int(np.sum(useful[flag_idx]))
    return {
        "flags": len(flag_idx),
        "recall": f"{detected}/{len(episodes)}",
        "precision": tp / len(flag_idx) if len(flag_idx) else float("nan"),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
        "loss_ahead": float(np.mean(loss_ahead)) if loss_ahead else float("nan"),
        "caught": len(loss_ahead),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "cpp/build/qtda_stream"))
    ap.add_argument("--csv", default=str(ROOT / "data/sp500.csv"))
    ap.add_argument("--dim", type=int, default=0, choices=[0, 1],
                    help="Betti dimension for the TDA signal (0=components, 1=loops)")
    ap.add_argument("--landscape", action="store_true",
                    help="basket mode: flag when the causal z-score of the H1 "
                         "landscape L1 norm level exceeds Z (expects a "
                         "multi-series CSV; events anchor on its first series)")
    args = ap.parse_args()

    if args.landscape and args.csv.endswith("sp500.csv"):
        args.csv = str(ROOT / "data/indices.csv")

    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"], dayfirst=not args.landscape)
    price_col = df.columns[1] if args.landscape else "close"
    prices = df[price_col].to_numpy(dtype=float)

    # TDA flags from the C++ engine, mapped back to price-series indices.
    if args.landscape:
        cmd = [args.binary, "--csv", args.csv, "--basket", "--landscape",
               "--eps", "1.0", "--w", "50"]
    else:
        cmd = [args.binary, "--csv", args.csv, "--dim", str(args.dim)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cpp = pd.read_csv(io.StringIO(proc.stdout))
    cpp["date"] = pd.to_datetime(cpp["date"], dayfirst=not args.landscape)
    date_to_idx = {d: i for i, d in enumerate(df["date"])}
    cpp_idx = cpp["date"].map(date_to_idx).to_numpy()
    if args.landscape:
        # Level, not jump: an elevated norm is the hypothesized precursor.
        # Two causal normalizations, both reported: expanding (all history)
        # and rolling one-year (local anomaly).
        norm = cpp["l1norm"].to_numpy(dtype=float)
        tda_exp = cpp_idx[causal_z(norm, min_hist=252) > Z]
        tda_flags = cpp_idx[np.nan_to_num(rolling_z(norm), nan=-np.inf) > Z]
        tda_label = "Landscape L1 roll-252 z>2"
    else:
        tda_exp = None
        tda_flags = cpp_idx[cpp["spike"].to_numpy() == 1]
        tda_label = f"TDA B{args.dim} spikes (z>2)"

    # Volatility baseline on the same dates.
    logret = np.diff(np.log(prices), prepend=np.nan)
    vol = pd.Series(logret).rolling(VOL_WINDOW).std().to_numpy()
    vol_z = causal_z(vol)
    # Restrict to the range where the TDA detector is also live, for fairness.
    live_start = cpp_idx[0]
    evaluable = np.arange(live_start, len(prices))
    vol_flags = evaluable[vol_z[evaluable] > Z]

    n_tda = len(tda_flags)
    order = evaluable[np.argsort(-np.nan_to_num(vol_z[evaluable], nan=-np.inf))]
    vol_matched = np.sort(order[:n_tda])

    episodes = [ep for ep in crash_episodes(prices) if ep["onset"] >= live_start]
    print(f"[events] {len(episodes)} crash episodes (drawdown < {DRAWDOWN:.0%}):")
    for ep in episodes:
        dd = prices[ep["trough"]] / prices[ep["peak"]] - 1.0
        print(f"  onset {df['date'].iloc[ep['onset']].date()}  "
              f"trough {df['date'].iloc[ep['trough']].date()}  ({dd:.0%})")

    rows = {tda_label: evaluate(tda_flags, episodes, prices)}
    if tda_exp is not None:
        rows["Landscape L1 expand z>2"] = evaluate(tda_exp, episodes, prices)
        # Give the volatility baseline the identical rolling normalization.
        vol_roll = evaluable[np.nan_to_num(rolling_z(vol)[evaluable], nan=-np.inf) > Z]
        rows[f"Vol roll-252 z>2 ({VOL_WINDOW}d)"] = evaluate(vol_roll, episodes, prices)
    rows[f"Vol z>2 ({VOL_WINDOW}d)"] = evaluate(vol_flags, episodes, prices)
    rows[f"Vol top-{n_tda} (matched)"] = evaluate(vol_matched, episodes, prices)
    print(f"\n{'detector':<26}{'flags':>7}{'recall':>9}{'precision':>11}{'med lead':>10}"
          f"{'loss ahead':>12}")
    for name, r in rows.items():
        lead = f"{r['median_lead']:.0f}d" if not np.isnan(r["median_lead"]) else "-"
        la = (f"{r['loss_ahead']:.0%} ({r['caught']}ep)"
              if not np.isnan(r["loss_ahead"]) else "-")
        print(f"{name:<26}{r['flags']:>7}{r['recall']:>9}{r['precision']:>11.2f}"
              f"{lead:>10}{la:>12}")
    print(f"\n(recall/lead over {PRE_WINDOW}d pre-onset window; precision counts flags "
          f"within [onset-{PRE_WINDOW}d, onset+{POST_WINDOW}d]; 'loss ahead' = share of "
          f"the episode's\npeak-to-trough loss still ahead at the first in-episode flag; "
          f"all detectors causal)")


if __name__ == "__main__":
    main()
