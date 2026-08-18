#pragma once
#include <cstddef>
#include <cstdint>
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

}  // namespace qtda
