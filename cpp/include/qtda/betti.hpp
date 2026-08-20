#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "union_find.hpp"

namespace qtda {

// Edge of the Vietoris-Rips filtration. a/b are stable point ids that survive
// window slides; SlotFn maps them to dense indices [0, n) for the union-find.
struct Edge {
  double dist;
  std::uint32_t a, b;
};

inline bool edge_less(const Edge& x, const Edge& y) { return x.dist < y.dist; }

// Betti-0 at every eps in eps_sorted (ascending): the number of connected
// components of the graph whose edges have length <= eps. One union-find
// sweep over the edges (ascending by dist) yields the whole curve.
//
// This matches ripser's H0 semantics as used in run_qtda.py: a component
// counted at eps iff its death (merge distance) is strictly greater than eps.
template <typename SlotFn>
inline void betti0_sweep(const std::vector<Edge>& edges_sorted, std::size_t n,
                         const std::vector<double>& eps_sorted, SlotFn slot_of,
                         UnionFind& uf, std::vector<int>& out) {
  uf.reset(n);
  std::size_t comps = n;
  std::size_t gi = 0;
  out.assign(eps_sorted.size(), 0);
  for (const Edge& e : edges_sorted) {
    while (gi < eps_sorted.size() && e.dist > eps_sorted[gi])
      out[gi++] = static_cast<int>(comps);
    if (gi == eps_sorted.size()) return;  // remaining merges can't affect the curve
    if (uf.unite(slot_of(e.a), slot_of(e.b))) --comps;
  }
  while (gi < eps_sorted.size()) out[gi++] = static_cast<int>(comps);
}

// Betti-1 at every eps in eps_sorted (ascending): the number of independent
// loops in the Vietoris-Rips complex at that scale. Standard Z/2 persistence
// reduction of the triangle boundary matrix, streamed in filtration order:
// a cycle is born when an edge closes a loop (union-find detects it), and dies
// at the diameter of the triangle whose reduced boundary column pivots on it.
// h1_persistence returns the paired intervals plus births still open at the
// filtration cap; betti1_sweep and landscape_l1 are both derived from it.
//
// `edges` must be sorted ascending by dist, pre-filtered to dist <= the cap
// (the largest grid eps; triangles past it can only kill loops at scales the
// grid never asks about), with a/b already dense slot indices in [0, n).
inline void h1_persistence(const std::vector<Edge>& edges, std::size_t n, UnionFind& uf,
                           std::vector<std::pair<double, double>>& pairs,
                           std::vector<double>& open_births) {
  pairs.clear();
  open_births.clear();
  const std::size_t ne = edges.size();
  // Edge-rank lookup: rank[x*n+y] = filtration index of edge (x,y), filled as
  // the sweep passes each edge, so a non-negative entry implies smaller rank.
  std::vector<std::int32_t> rank(n * n, -1);
  auto rk = [&rank, n](std::uint32_t x, std::uint32_t y) -> std::int32_t& {
    return rank[x * n + y];
  };
  uf.reset(n);
  std::vector<char> positive(ne, 0);              // cycle-creating edges
  std::vector<std::vector<std::int32_t>> nb(n);   // neighbors via smaller-rank edges
  std::vector<std::int32_t> owner(ne, -1);        // pivot edge rank -> reduced column
  std::vector<std::vector<std::int32_t>> cols;    // stored reduced columns
  std::vector<std::int32_t> col, tmp;
  for (std::size_t r = 0; r < ne; ++r) {
    const std::uint32_t a = edges[r].a, b = edges[r].b;
    if (!uf.unite(a, b)) positive[r] = 1;
    // Triangles closed by this edge (their max-rank edge): common neighbors of
    // a and b already connected to both at smaller ranks.
    const bool a_small = nb[a].size() <= nb[b].size();
    const std::vector<std::int32_t>& small = a_small ? nb[a] : nb[b];
    const std::uint32_t other = a_small ? b : a;
    for (std::int32_t c : small) {
      if (rk(other, static_cast<std::uint32_t>(c)) < 0) continue;
      std::int32_t r1 = rk(a, static_cast<std::uint32_t>(c));
      std::int32_t r2 = rk(b, static_cast<std::uint32_t>(c));
      if (r1 > r2) std::swap(r1, r2);
      col.clear();
      col.push_back(r1);
      col.push_back(r2);
      col.push_back(static_cast<std::int32_t>(r));
      while (!col.empty()) {  // reduce against earlier columns sharing the pivot
        const std::int32_t own = owner[static_cast<std::size_t>(col.back())];
        if (own < 0) break;
        tmp.clear();
        std::set_symmetric_difference(col.begin(), col.end(),
                                      cols[static_cast<std::size_t>(own)].begin(),
                                      cols[static_cast<std::size_t>(own)].end(),
                                      std::back_inserter(tmp));
        col.swap(tmp);
      }
      if (!col.empty()) {
        owner[static_cast<std::size_t>(col.back())] = static_cast<std::int32_t>(cols.size());
        // Loop born at the pivot edge's diameter dies at this triangle's.
        pairs.emplace_back(edges[static_cast<std::size_t>(col.back())].dist, edges[r].dist);
        cols.push_back(col);
      }
    }
    nb[a].push_back(static_cast<std::int32_t>(b));
    nb[b].push_back(static_cast<std::int32_t>(a));
    rk(a, b) = rk(b, a) = static_cast<std::int32_t>(r);
  }
  // Cycle-creating edges never paired by a triangle: loops still open at the cap.
  for (std::size_t r = 0; r < ne; ++r)
    if (positive[r] && owner[r] < 0) open_births.push_back(edges[r].dist);
}

inline void betti1_sweep(const std::vector<Edge>& edges, std::size_t n,
                         const std::vector<double>& eps_sorted, UnionFind& uf,
                         std::vector<int>& out) {
  std::vector<std::pair<double, double>> pairs;
  std::vector<double> births;  // reused as open births, then extended to all births
  h1_persistence(edges, n, uf, pairs, births);
  std::vector<double> deaths;
  deaths.reserve(pairs.size());
  for (const auto& p : pairs) {
    births.push_back(p.first);
    deaths.push_back(p.second);
  }
  std::sort(births.begin(), births.end());
  std::sort(deaths.begin(), deaths.end());
  out.assign(eps_sorted.size(), 0);
  for (std::size_t gi = 0; gi < eps_sorted.size(); ++gi) {
    const double e = eps_sorted[gi];
    const auto born = std::upper_bound(births.begin(), births.end(), e) - births.begin();
    const auto died = std::upper_bound(deaths.begin(), deaths.end(), e) - deaths.begin();
    out[gi] = static_cast<int>(born - died);
  }
}

// L1 norm of the H1 persistence landscape, computed exactly: the landscape
// layers are a pointwise re-sorting of the interval tent functions, so the sum
// over layers integrates to sum of (death - birth)^2 / 4 over intervals. Loops
// still open at the cap are truncated there (the Gidea-Katz convention).
inline double landscape_l1(const std::vector<std::pair<double, double>>& pairs,
                           const std::vector<double>& open_births, double cap) {
  double s = 0.0;
  for (const auto& p : pairs) {
    const double len = p.second - p.first;
    s += len * len;
  }
  for (double b : open_births) {
    const double len = cap - b;
    s += len * len;
  }
  return s / 4.0;
}

}  // namespace qtda
