# Quantum TDA for Financial Bubble Detection  

## Overview  
Financial markets can look stable until they hit a tipping point: a **bubble** or **crash**.  
While traditional indicators (volatility, moving averages, etc.) often lag, **Topological Data Analysis (TDA)** captures deeper geometric changes in the market’s structure.  

This repository applies **classical TDA** and other quantum methods to S&P 500 data to identify when the “shape” of the market changes in ways consistent with bubbles.  

---

## Pipeline  
- **Log transformation** of prices for numerical stability.  
- **Takens embedding** to reconstruct the market’s phase space.  
- **Sliding windows** to analyze evolving structure over time.  
- **Betti₀ curves** across multiple ε-scales to measure fragmentation.  
- **Pairwise L² deltas** to quantify sudden topological shifts.  

---

## Files  
- `src/takens.py` – Builds Takens embeddings and sliding windows from time series  
- `src/betti_curves.py` – Computes Betti curves, Lᵖ deltas, and statistical spike detection  
- `src/rips_laplacian.py` – Constructs Vietoris–Rips complex and Laplacian for quantum demo  
- `src/qpe.py` –  Demonstrates Quantum Phase Estimation (QPE) using Qiskit 
- `scripts/run_qpe.py` – QPE demo on one window/ε (bar chart comparing classical vs QPE Betti) 
- `scripts/run_qtda.py` – Main script to run the full analysis  
- `scripts/validate_cpp.py` – Cross-checks the C++ engine against ripser ground truth  
- `scripts/live_feed.py` – Live exchange WebSocket feed (Coinbase/Binance) for piping into the engine  
- `scripts/backtest_baseline.py` – Causal backtest: TDA spikes vs. a realized-volatility baseline  
- `cpp/` – C++17 streaming engine (see **C++ Streaming Engine** below)  
- `data/sp500.csv` – Daily S&P 500 closing prices (date, value)  
- `plots/betti_curves.png` – Betti₀ curves over time  
- `plots/crash_spikes.png` – Pairwise L² deltas with spikes flagged  
- `plots/qpe_betti_k*_eps*_*.png` – QPE bar chart (filename includes window’s center date)


---

## Results & Interpretation

### 1. Betti Curves over Windows
Tracks the number of connected components (**Betti₀**) for each sliding window at several neighborhood radii (ε).  
**Axes:**  
- **X:** Calendar date (center of each window)  
- **Y:** Betti₀ (connectivity)

**How to read:**  
- Values near **1** show cohesive, stable market behavior.  
- **Spikes** show fragmentation in the geometry, consistent with instability or bubble formation.  
These spikes often precede major events (e.g., 2008, 2020).

---

### 2. Pairwise L² Deltas (Spikes Flagged)
Measures how abruptly the topology changes between consecutive windows via the L² distance between their Betti curves.  
**Axes:**  
- **X:** Date (of the later window)  
- **Y:** Δ (L² distance between consecutive Betti curves)

**How to read:**  
- Taller bars indicate larger step-changes in topology.  
- **“×” markers** denote statistically significant spikes (z-score > 2.0).  
When **Betti₀ spikes** and **Δ spikes** occur together, they strongly align with **bubble bursts**.

---

### 3. Quantum Betti Estimation (QPE)
Builds a Vietoris–Rips Laplacian for one window and uses **Quantum Phase Estimation (QPE)** on a simulator to estimate the Betti number from the Laplacian’s spectrum (zero eigenvalues ↔ connected components).

**What the QPE bar chart shows**  
- **Classical** (left): Betti₀ from persistent homology (ripser) — the true connectivity for that window.  
- **QPE est.** (right): Betti₀ estimated from the QPE zero-phase probability.

**How to interpret:**  
- The **classical** Betti₀ reflects the actual connectivity in that window.  
- The **QPE estimate** is an approximation; with few phase qubits and limited shots it can overestimate, but it illustrates how topological information can be extracted from the Laplacian using a quantum routine.


---

## Accuracy & Insight  
- Detects early fragmentation during the **2007–2008** financial crisis.  
- Shows a sharp topological shift around **March 2020** (COVID-19 crash).  
These findings demonstrate that TDA-based indicators can act as **early signals of market regime change**, often preceding traditional metrics.  

---

## C++ Streaming Engine (`cpp/`)

A real-time reimplementation of the TDA pipeline as a C++17 streaming engine: feed it one price per tick and it emits the Betti₀ curve, topology delta, and spike flag for the window ending at that tick — causally, with no lookahead.

**How it works**
- Each tick appends a log price, completes a new Takens embedding point, and slides the window.
- **Incremental filtration update:** only the `w−1` distances involving the new point are computed (`O(w·m)` instead of `O(w²·m)`), then merged into an always-sorted edge list (`O(E)` merge instead of an `O(E log E)` re-sort).
- **Single-sweep Betti₀:** one union-find pass (path halving + union by size) over the sorted edges yields the component count at every ε threshold simultaneously — no per-ε recomputation, no persistent homology library.
- Spike detection uses a running (Welford) z-score over the L² deltas, so the signal is fully causal and suitable for live monitoring.

**Build & run**
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
./cpp/build/qtda_stream --csv data/sp500.csv            # stream analysis to stdout
./cpp/build/qtda_stream --csv - < data/sp500.csv        # or pipe prices via stdin
./cpp/build/qtda_stream --csv data/sp500.csv --benchmark
```
Output CSV: `date,betti@eps=…,delta,zscore,spike`.

**Performance** (Apple M-series, single core, full S&P 500 series)

| Window | Edges | Incremental (mean/tick) | Full recompute | Speedup |
|--------|-------|------------------------|----------------|---------|
| w=50   | 1,225 | 8.1 µs | 42.2 µs | 5.2× |
| w=250  | 31,125 | 157 µs | 1,443 µs | 9.2× |

**Correctness**
- `--benchmark` cross-checks the incremental edge list against a from-scratch recompute every tick (0 mismatches over all windows).
- `scripts/validate_cpp.py` compares the engine's Betti₀ curves to ripser's persistent-homology output: **all 5,472 windows match exactly** (`--full`), and the delta column is bit-consistent with the emitted curves.
- The causal spike flags concentrate in 2002, 2008–09, and 2020 — the dot-com bear market, the financial crisis, and the COVID crash — consistent with the batch pipeline's findings.

### Live feed demo

The engine is fully causal, so it can run on a live market. `scripts/live_feed.py` connects to a public exchange WebSocket (Coinbase by default, Binance optional, no API key) and pipes one sampled price per interval into the engine's stdin:

```bash
python scripts/live_feed.py --product BTC-USD --interval 1.0 \
  | ./cpp/build/qtda_stream --csv - --w 30 --eps auto
```

`--eps auto` calibrates the ε grid from distance quantiles (25/50/75/90%) of the first full window, so the same binary works at any price scale and frequency — on daily S&P closes auto-calibration recovers ε ≈ 0.055/0.076/0.102/0.128, nearly identical to the hand-tuned grid; on 1-second BTC ticks it lands around 10⁻⁴. The feed reconnects automatically on drops.

### Does it beat a volatility filter? (honest backtest)

`scripts/backtest_baseline.py` evaluates the causal TDA spikes against a 20-day realized-volatility z-score on identical terms: same dates, same z>2 threshold (plus a flag-count-matched variant), against objectively defined crash episodes (drawdown from trailing peak crossing −15%; 2000, 2008, 2018, 2020 in this dataset).

| Detector | Flags | Pre-onset recall (60d) | Precision | Loss still ahead at first flag |
|----------|-------|------------------------|-----------|-------------------------------|
| TDA spikes (z>2) | 125 | 0/4 | 0.25 | 35% (3 episodes) |
| Vol z>2 (20d) | 166 | 1/4 | 0.27 | 57% (3 episodes) |
| Vol matched (top-125) | 125 | 1/4 | 0.30 | 43% (3 episodes) |

**Honest takeaway:** under causal evaluation, the Betti₀ spike signal is a *coincident* crash detector, not a leading indicator — it fires during the violent phase of drawdowns (e.g. from 6 Oct 2008, with ~35% of the eventual peak-to-trough loss still ahead) but not in the 60 trading days before onset, and on this dataset it does not beat a simple volatility filter for early warning. The apparent "spikes precede crashes" pattern in the batch plots comes from the batch z-score's use of full-sample statistics, which a live signal doesn't have. Making the topological signal genuinely leading (e.g. via Betti₁ loop structure) is an open direction.

---

## Credit  
This work was **inspired by the Moody’s Quantum Challenge at iQuHACK 2025**, which encouraged exploration of quantum and topological methods for financial risk analysis.   

---

## License  
MIT License © 2025
