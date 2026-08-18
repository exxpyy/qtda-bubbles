#pragma once
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace qtda {

// Disjoint-set with path halving and union by size.
class UnionFind {
 public:
  void reset(std::size_t n) {
    parent_.resize(n);
    std::iota(parent_.begin(), parent_.end(), 0u);
    size_.assign(n, 1u);
  }

  std::uint32_t find(std::uint32_t x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];  // path halving
      x = parent_[x];
    }
    return x;
  }

  // Returns true if a and b were in different components (a merge happened).
  bool unite(std::uint32_t a, std::uint32_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (size_[a] < size_[b]) std::swap(a, b);
    parent_[b] = a;
    size_[a] += size_[b];
    return true;
  }

 private:
  std::vector<std::uint32_t> parent_;
  std::vector<std::uint32_t> size_;
};

}  // namespace qtda
