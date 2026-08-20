#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

#include "betti.hpp"
#include "union_find.hpp"

namespace qtda {

struct DetectorConfig {
  int m = 4;                                           // Takens embedding dimension
  int d = 5;                                           // Takens delay
  int w = 50;                                          // window size (embedding points)
  int dim = 0;                                         // Betti dimension: 0 = components,
                                                       // 1 = loops (persistence reduction)
  bool landscape = false;                              // emit the H1 landscape L1 norm
                                                       // instead of Betti counts (implies
                                                       // dim 1; eps.back() is the cap)
  std::vector<double> eps = {0.05, 0.07, 0.10, 0.12};  // sorted ascending on construction;
                                                       // empty = auto-calibrate from the
                                                       // first full window (see below)
  double z = 2.0;                                      // spike threshold on delta z-score
  int min_history = 10;                                // deltas required before z-scores emit
};

struct Signal {
  bool ready = false;      // full window available; betti (or l1norm) is valid
  std::vector<int> betti;  // Betti counts per eps (empty in landscape mode)
  double l1norm = 0.0;     // H1 landscape L1 norm (landscape mode only)
  bool has_norm = false;
  bool has_delta = false;  // a previous curve existed
  double delta = 0.0;      // L2 distance to the previous window's curve
  bool has_z = false;      // enough delta history for a z-score
  double zscore = 0.0;
  bool spike = false;      // zscore > cfg.z
};

// Streaming bubble detector: feed one price per tick, get the Betti-0 curve,
// topology delta, and spike flag for the window ending at that tick.
//
// Incremental design: when the window slides by one embedding point, only the
// w-1 distances involving the new point are computed (O(w*m) work) and merged
// into the always-sorted edge list (O(E) merge). A full rebuild would cost
// O(w^2*m) distance work plus an O(E log E) sort per tick; see betti_naive().
class StreamingDetector {
 public:
  explicit StreamingDetector(DetectorConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.m < 1 || cfg_.d < 1 || cfg_.w < 2)
      throw std::invalid_argument("require m >= 1, d >= 1, w >= 2");
    if (cfg_.dim != 0 && cfg_.dim != 1)
      throw std::invalid_argument("dim must be 0 or 1");
    if (cfg_.landscape) cfg_.dim = 1;
    std::sort(cfg_.eps.begin(), cfg_.eps.end());
    lag_ = cfg_.d * (cfg_.m - 1);
    const std::size_t e = static_cast<std::size_t>(cfg_.w) * (cfg_.w - 1) / 2;
    edges_.reserve(e + cfg_.w);
    scratch_.reserve(cfg_.w);
  }

  const DetectorConfig& config() const { return cfg_; }

  Signal on_price(double price) { return on_log_price(std::log(price)); }

  Signal on_log_price(double x) {
    xbuf_.push_back(x);
    if (static_cast<int>(xbuf_.size()) > lag_ + 1) xbuf_.pop_front();
    if (static_cast<int>(xbuf_.size()) < lag_ + 1) return Signal{};

    // The embedding point completed by this tick, oldest coordinate first:
    // (x[t-d(m-1)], x[t-d(m-2)], ..., x[t]) — same rows as takens_embedding().
    std::vector<double> p(cfg_.m);
    for (int k = 0; k < cfg_.m; ++k) p[static_cast<std::size_t>(k)] = xbuf_[static_cast<std::size_t>(k * cfg_.d)];
    return step(std::move(p));
  }

  // Basket mode: one tick carries one price per series, and the point-cloud
  // point is the vector of log returns across the basket (Gidea-Katz style),
  // so cross-series co-movement shapes the geometry directly. Takens m/d are
  // not used.
  Signal on_prices(const std::vector<double>& prices) {
    std::vector<double> logs(prices.size());
    for (std::size_t i = 0; i < prices.size(); ++i) logs[i] = std::log(prices[i]);
    if (prev_logs_.empty()) {
      prev_logs_ = std::move(logs);
      return Signal{};
    }
    if (logs.size() != prev_logs_.size())
      throw std::invalid_argument("basket width changed mid-stream");
    std::vector<double> r(logs.size());
    for (std::size_t i = 0; i < logs.size(); ++i) r[i] = logs[i] - prev_logs_[i];
    prev_logs_ = std::move(logs);
    return step(std::move(r));
  }

  // Shared per-tick path: append the point, slide the window, compute the
  // configured statistic once the window is full.
  Signal step(std::vector<double> p) {
    add_point(std::move(p));
    Signal sig;
    if (static_cast<int>(points_.size()) < cfg_.w) return sig;

    // Auto-calibration: with an empty eps grid, pick distance quantiles of the
    // first full window so the scale adapts to any asset/frequency. edges_ is
    // already sorted, so quantiles are direct lookups.
    if (cfg_.eps.empty()) {
      const double quantiles[] = {0.25, 0.50, 0.75, 0.90};
      const std::size_t last = edges_.size() - 1;
      for (double q : quantiles)
        cfg_.eps.push_back(edges_[static_cast<std::size_t>(q * static_cast<double>(last))].dist);
    }

    sig.ready = true;
    const std::uint32_t base = base_id_;
    if (cfg_.dim == 0) {
      betti0_sweep(edges_, points_.size(), cfg_.eps,
                   [base](std::uint32_t id) { return id - base; }, uf_, sig.betti);
    } else {
      // Translate stable ids to dense slots and cap at the grid maximum;
      // edges_ is sorted, so stop at the first edge past it.
      capped_.clear();
      for (const Edge& e : edges_) {
        if (e.dist > cfg_.eps.back()) break;
        capped_.push_back({e.dist, e.a - base, e.b - base});
      }
      if (cfg_.landscape) {
        h1_persistence(capped_, points_.size(), uf_, pairs_, open_births_);
        sig.l1norm = landscape_l1(pairs_, open_births_, cfg_.eps.back());
        sig.has_norm = true;
      } else {
        betti1_sweep(capped_, points_.size(), cfg_.eps, uf_, sig.betti);
      }
    }

    bool have_delta = false;
    if (cfg_.landscape) {
      if (has_prev_norm_) {
        sig.delta = std::abs(sig.l1norm - prev_norm_);
        have_delta = true;
      }
      prev_norm_ = sig.l1norm;
      has_prev_norm_ = true;
    } else {
      if (!prev_curve_.empty()) {
        double s = 0.0;
        for (std::size_t j = 0; j < sig.betti.size(); ++j) {
          const double diff = sig.betti[j] - prev_curve_[j];
          s += diff * diff;
        }
        sig.delta = std::sqrt(s);
        have_delta = true;
      }
      prev_curve_ = sig.betti;
    }

    if (have_delta) {
      sig.has_delta = true;
      // Running (causal) z-score via Welford; population sd like numpy .std().
      ++delta_count_;
      const double d1 = sig.delta - delta_mean_;
      delta_mean_ += d1 / static_cast<double>(delta_count_);
      delta_m2_ += d1 * (sig.delta - delta_mean_);
      if (delta_count_ >= static_cast<std::size_t>(cfg_.min_history)) {
        const double sd = std::sqrt(delta_m2_ / static_cast<double>(delta_count_));
        sig.has_z = true;
        sig.zscore = (sig.delta - delta_mean_) / (sd + 1e-9);
        sig.spike = sig.zscore > cfg_.z;
      }
    }
    return sig;
  }

  // Full recompute of the current window's Betti-0 curve from scratch:
  // all-pairs distances, full sort, sweep. Used as the benchmark baseline and
  // as a correctness cross-check against the incremental edge list.
  std::vector<int> betti_naive() const {
    const std::size_t n = points_.size();
    std::vector<Edge> edges;
    edges.reserve(n * (n - 1) / 2);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < points_[i].size(); ++k) {
          const double diff = points_[i][k] - points_[j][k];
          s += diff * diff;
        }
        edges.push_back({std::sqrt(s), static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j)});
      }
    }
    std::sort(edges.begin(), edges.end(), edge_less);
    UnionFind uf;
    std::vector<int> out;
    if (cfg_.dim == 0) {
      betti0_sweep(edges, n, cfg_.eps, [](std::uint32_t id) { return id; }, uf, out);
    } else {
      const auto past = std::upper_bound(edges.begin(), edges.end(),
                                         Edge{cfg_.eps.back(), 0, 0}, edge_less);
      edges.erase(past, edges.end());
      betti1_sweep(edges, n, cfg_.eps, uf, out);
    }
    return out;
  }

 private:
  void add_point(std::vector<double> p) {
    if (static_cast<int>(points_.size()) == cfg_.w) {
      // Evict the oldest point and every edge incident to it (one linear pass;
      // the surviving edges stay sorted).
      const std::uint32_t old_id = base_id_;
      points_.pop_front();
      ++base_id_;
      edges_.erase(std::remove_if(edges_.begin(), edges_.end(),
                                  [old_id](const Edge& e) { return e.a == old_id || e.b == old_id; }),
                   edges_.end());
    }

    const std::uint32_t id = next_id_++;
    scratch_.clear();
    for (std::size_t i = 0; i < points_.size(); ++i) {
      double s = 0.0;
      for (std::size_t k = 0; k < p.size(); ++k) {
        const double diff = points_[i][k] - p[k];
        s += diff * diff;
      }
      scratch_.push_back({std::sqrt(s), base_id_ + static_cast<std::uint32_t>(i), id});
    }
    std::sort(scratch_.begin(), scratch_.end(), edge_less);
    const std::size_t mid = edges_.size();
    edges_.insert(edges_.end(), scratch_.begin(), scratch_.end());
    std::inplace_merge(edges_.begin(), edges_.begin() + static_cast<std::ptrdiff_t>(mid), edges_.end(), edge_less);

    points_.push_back(std::move(p));
  }

  DetectorConfig cfg_;
  int lag_ = 0;  // d*(m-1): prices needed before the first embedding point

  std::deque<double> xbuf_;                 // last lag_+1 log prices
  std::deque<std::vector<double>> points_;  // current window, oldest first
  std::uint32_t next_id_ = 0;               // id of the next embedding point
  std::uint32_t base_id_ = 0;               // id of the oldest surviving point
  std::vector<Edge> edges_;                 // all window edges, sorted by dist
  std::vector<Edge> scratch_;               // new point's edges, per tick
  std::vector<Edge> capped_;                // slot-indexed edges <= eps max (dim 1)
  UnionFind uf_;

  std::vector<double> prev_logs_;  // basket mode: previous tick's log prices
  std::vector<std::pair<double, double>> pairs_;  // landscape mode scratch
  std::vector<double> open_births_;

  std::vector<int> prev_curve_;
  double prev_norm_ = 0.0;  // landscape mode: previous window's L1 norm
  bool has_prev_norm_ = false;
  std::size_t delta_count_ = 0;  // Welford running stats over deltas
  double delta_mean_ = 0.0;
  double delta_m2_ = 0.0;
};

}  // namespace qtda
