"""Cross-check the C++ streaming engine (cpp/qtda_stream) against ripser.

Recomputes Betti-0 curves for sampled windows with the same pipeline as
run_qtda.py (log prices -> Takens embedding -> sliding windows -> ripser H0,
counting intervals with birth < eps < death) and compares them to the C++
engine's output row-for-row. Also checks the delta column is consistent with
the emitted Betti curves.

Usage:
    python scripts/validate_cpp.py [--limit 500] [--full]
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.takens import takens_embedding, sliding_windows  # noqa: E402
from ripser import ripser  # noqa: E402


def betti_ripser(pc: np.ndarray, eps_grid, dim: int) -> list[int]:
    dgm = ripser(pc, maxdim=dim)["dgms"][dim]
    # H0 births are all 0 so strictness of the birth comparison is moot there;
    # for H1 the complex at eps includes simplices of diameter exactly eps,
    # hence birth <= eps.
    return [int(np.sum((dgm[:, 0] <= e) & (e < dgm[:, 1]))) for e in eps_grid]


def validate_landscape(args) -> int:
    """Check basket-mode H1 landscape L1 norms against ripser: the norm is
    sum of (death - birth)^2 / 4 over the H1 diagram of each window of
    cross-series log-return vectors (uncapped filtration, so all loops die
    naturally and the norm is parameter-free)."""
    cmd = [args.binary, "--csv", args.csv, "--basket", "--landscape",
           "--eps", "1.0", "--w", str(args.w)]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cpp = pd.read_csv(io.StringIO(proc.stdout))

    df = pd.read_csv(args.csv, index_col=0)
    X = np.diff(np.log(df.to_numpy(dtype=float)), axis=0)
    windows = sliding_windows(X, w=args.w)
    assert len(windows) == len(cpp), (
        f"window count mismatch: python={len(windows)} cpp={len(cpp)}")
    print(f"[ok] window count matches: {len(windows)}")

    idx = np.unique(np.linspace(0, len(windows) - 1, args.limit).astype(int))
    print(f"[run] checking {len(idx)} windows against ripser ...")
    cpp_norm = cpp["l1norm"].to_numpy(dtype=float)
    mismatches = 0
    for n, i in enumerate(idx):
        dgm = ripser(windows[i], maxdim=1)["dgms"][1]
        ref = float(np.sum((dgm[:, 1] - dgm[:, 0]) ** 2) / 4.0)
        # ripser emits float32 diagrams; compare with matching tolerance.
        if not np.isclose(ref, cpp_norm[i], rtol=1e-4, atol=1e-15):
            mismatches += 1
            print(f"[FAIL] window {i} ({cpp.iloc[i, 0]}): ripser={ref:.6e} "
                  f"cpp={cpp_norm[i]:.6e}")
        if (n + 1) % 100 == 0:
            print(f"  ... {n + 1}/{len(idx)}")
    if mismatches == 0:
        print(f"[PASS] {len(idx)} windows: landscape norms match ripser")
        return 0
    print(f"[FAIL] {mismatches} mismatching windows")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "cpp/build/qtda_stream"))
    ap.add_argument("--csv", default=str(ROOT / "data/sp500.csv"))
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--w", type=int, default=50)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.07, 0.1, 0.12])
    ap.add_argument("--dim", type=int, default=0, choices=[0, 1])
    ap.add_argument("--landscape", action="store_true",
                    help="validate basket-mode H1 landscape L1 norms on a "
                         "multi-series CSV (pass it via --csv) against ripser")
    ap.add_argument("--limit", type=int, default=500,
                    help="number of windows to check against ripser (evenly sampled)")
    ap.add_argument("--full", action="store_true", help="check every window")
    args = ap.parse_args()

    if args.landscape:
        return validate_landscape(args)

    eps = sorted(args.eps)
    cmd = [args.binary, "--csv", args.csv, "--m", str(args.m), "--d", str(args.d),
           "--w", str(args.w), "--dim", str(args.dim),
           "--eps", ",".join(str(e) for e in eps)]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cpp = pd.read_csv(io.StringIO(proc.stdout))
    betti_cols = [c for c in cpp.columns if c.startswith("betti@")]
    cpp_betti = cpp[betti_cols].to_numpy(dtype=int)

    prices = pd.read_csv(args.csv)["close"].to_numpy(dtype=float)
    x = np.log(prices)
    X = takens_embedding(x, m=args.m, d=args.d)
    windows = sliding_windows(X, w=args.w)
    assert len(windows) == len(cpp), (
        f"window count mismatch: python={len(windows)} cpp={len(cpp)}")
    print(f"[ok] window count matches: {len(windows)}")

    if args.full:
        idx = np.arange(len(windows))
    else:
        idx = np.unique(np.linspace(0, len(windows) - 1, args.limit).astype(int))
    print(f"[run] checking {len(idx)} windows against ripser ...")

    mismatches = 0
    for n, i in enumerate(idx):
        ref = betti_ripser(windows[i], eps, args.dim)
        got = cpp_betti[i].tolist()
        if ref != got:
            mismatches += 1
            print(f"[FAIL] window {i} ({cpp.iloc[i, 0]}): ripser={ref} cpp={got}")
        if (n + 1) % 100 == 0:
            print(f"  ... {n + 1}/{len(idx)}")

    # Delta column must equal the L2 diff of consecutive emitted curves.
    deltas = np.linalg.norm(np.diff(cpp_betti, axis=0), axis=1)
    cpp_deltas = cpp["delta"].to_numpy(dtype=float)[1:]
    delta_ok = np.allclose(deltas, cpp_deltas, atol=1e-9)
    print(f"[{'ok' if delta_ok else 'FAIL'}] delta column consistent with Betti curves")

    if mismatches == 0 and delta_ok:
        print(f"[PASS] {len(idx)} windows match ripser exactly; deltas consistent")
        return 0
    print(f"[FAIL] {mismatches} mismatching windows")
    return 1


if __name__ == "__main__":
    sys.exit(main())
