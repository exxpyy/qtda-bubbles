// qtda_stream: streaming topological bubble detector.
//
// Reads a price series (CSV: date,price — or any file whose price column is
// named close/price or is the last column) from a file or stdin, and for each
// tick emits the Betti-0 curve of the sliding Takens-embedding window ending
// at that tick, the L2 delta vs. the previous window, and a causal z-score
// spike flag. Matches the conventions of scripts/run_qtda.py (log prices,
// right-aligned windows, ripser H0 semantics).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qtda/detector.hpp"

namespace {

using qtda::DetectorConfig;
using qtda::Signal;
using qtda::StreamingDetector;

std::vector<std::string> split(const std::string& line, char sep) {
  std::vector<std::string> out;
  std::string field;
  std::stringstream ss(line);
  while (std::getline(ss, field, sep)) out.push_back(field);
  return out;
}

bool parse_double(const std::string& s, double& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end != s.c_str() && *end == '\0';
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

int find_price_col(const std::vector<std::string>& header) {
  const std::vector<std::string> names = {"close", "adj close", "price", "value"};
  for (const std::string& name : names)
    for (std::size_t i = 0; i < header.size(); ++i)
      if (lower(header[i]) == name) return static_cast<int>(i);
  return -1;
}

struct BenchStats {
  std::vector<double> us;  // per-tick latency, microseconds
  void add(std::chrono::steady_clock::duration dt) {
    us.push_back(std::chrono::duration<double, std::micro>(dt).count());
  }
  double mean() const {
    double s = 0.0;
    for (double v : us) s += v;
    return us.empty() ? 0.0 : s / static_cast<double>(us.size());
  }
  double pct(double p) {
    if (us.empty()) return 0.0;
    std::sort(us.begin(), us.end());
    const std::size_t i = static_cast<std::size_t>(p * static_cast<double>(us.size() - 1));
    return us[i];
  }
};

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --csv PATH        input CSV, '-' for stdin (default: data/sp500.csv)\n"
      << "  --m N             Takens embedding dimension (default 4)\n"
      << "  --d N             Takens delay (default 5)\n"
      << "  --w N             sliding window size (default 50)\n"
      << "  --dim K           Betti dimension: 0 = components (default),\n"
      << "                    1 = loops via Z/2 persistence reduction\n"
      << "  --basket          multi-series input (date,p1,p2,...): each tick's\n"
      << "                    point is the vector of log returns across the\n"
      << "                    basket; Takens m/d are not used\n"
      << "  --landscape       emit the H1 persistence landscape L1 norm per\n"
      << "                    window instead of Betti counts (implies --dim 1;\n"
      << "                    the largest eps is the filtration cap)\n"
      << "  --eps a,b,c       epsilon grid (default 0.05,0.07,0.1,0.12), or 'auto'\n"
      << "                    to calibrate from distance quantiles (25/50/75/90%)\n"
      << "                    of the first full window — works at any price scale\n"
      << "  --z X             spike z-threshold (default 2.0)\n"
      << "  --min-history N   deltas required before z-scores emit (default 10)\n"
      << "  --benchmark       time incremental vs naive full-recompute per tick;\n"
      << "                    also cross-checks that both produce identical curves\n"
      << "Output CSV on stdout: date,betti@eps...,delta,zscore,spike\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv_path = "data/sp500.csv";
  DetectorConfig cfg;
  bool benchmark = false;
  bool basket = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "error: " << what << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--csv") {
      csv_path = next("--csv");
    } else if (arg == "--m") {
      cfg.m = std::stoi(next("--m"));
    } else if (arg == "--d") {
      cfg.d = std::stoi(next("--d"));
    } else if (arg == "--w") {
      cfg.w = std::stoi(next("--w"));
    } else if (arg == "--eps") {
      cfg.eps.clear();
      const std::string val = next("--eps");
      if (val != "auto") {
        for (const std::string& tok : split(val, ',')) {
          double v;
          if (!parse_double(tok, v)) {
            std::cerr << "error: bad eps value '" << tok << "'\n";
            return 2;
          }
          cfg.eps.push_back(v);
        }
      }
    } else if (arg == "--dim") {
      cfg.dim = std::stoi(next("--dim"));
    } else if (arg == "--basket") {
      basket = true;
    } else if (arg == "--landscape") {
      cfg.landscape = true;
    } else if (arg == "--z") {
      cfg.z = std::stod(next("--z"));
    } else if (arg == "--min-history") {
      cfg.min_history = std::stoi(next("--min-history"));
    } else if (arg == "--benchmark") {
      benchmark = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "error: unknown argument '" << arg << "'\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (benchmark && cfg.landscape) {
    std::cerr << "error: --benchmark compares Betti curves; not supported with --landscape\n";
    return 2;
  }
  StreamingDetector det(cfg);

  std::ifstream file;
  std::istream* in = &std::cin;
  if (csv_path != "-") {
    file.open(csv_path);
    if (!file) {
      std::cerr << "error: cannot open " << csv_path << "\n";
      return 1;
    }
    in = &file;
  }

  // With --eps auto the grid isn't known until the first full window, so the
  // header is deferred until then.
  bool header_written = false;
  auto write_header = [&det, &header_written]() {
    std::cout << "date";
    if (det.config().landscape) {
      std::cout << ",l1norm";
    } else {
      for (double e : det.config().eps) std::cout << ",betti@eps=" << e;
    }
    std::cout << ",delta,zscore,spike" << std::endl;
    header_written = true;
  };
  if (det.config().landscape || !det.config().eps.empty()) write_header();

  BenchStats inc_stats, naive_stats;
  std::size_t naive_mismatches = 0;
  std::size_t rows = 0, ticks = 0, spikes = 0;

  std::string line;
  bool first_line = true;
  int price_col = -1;  // -1: last column
  while (std::getline(*in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<std::string> fields = split(line, ',');
    if (first_line) {
      first_line = false;
      double probe;
      if (!parse_double(fields.back(), probe)) {  // header row
        price_col = find_price_col(fields);
        continue;
      }
    }
    Signal sig;
    std::chrono::steady_clock::time_point t0, t1;
    if (basket) {
      // All columns after the date are one price per series.
      std::vector<double> prices;
      prices.reserve(fields.size() - 1);
      bool ok = fields.size() >= 3;
      for (std::size_t i = 1; ok && i < fields.size(); ++i) {
        double v;
        ok = parse_double(fields[i], v) && v > 0.0;
        prices.push_back(v);
      }
      if (!ok) {
        std::cerr << "[warn] skipping unparseable row: " << line << "\n";
        continue;
      }
      t0 = std::chrono::steady_clock::now();
      sig = det.on_prices(prices);
      t1 = std::chrono::steady_clock::now();
    } else {
      const std::size_t col =
          price_col >= 0 ? static_cast<std::size_t>(price_col) : fields.size() - 1;
      double price;
      if (col >= fields.size() || !parse_double(fields[col], price) || price <= 0.0) {
        std::cerr << "[warn] skipping unparseable row: " << line << "\n";
        continue;
      }
      t0 = std::chrono::steady_clock::now();
      sig = det.on_price(price);
      t1 = std::chrono::steady_clock::now();
    }
    ++ticks;
    if (!sig.ready) continue;

    if (benchmark) {
      inc_stats.add(t1 - t0);
      const auto n0 = std::chrono::steady_clock::now();
      std::vector<int> ref = det.betti_naive();
      const auto n1 = std::chrono::steady_clock::now();
      naive_stats.add(n1 - n0);
      if (ref != sig.betti) ++naive_mismatches;
    }

    if (!header_written) write_header();
    std::cout << fields[0];
    if (sig.has_norm)
      std::cout << ',' << sig.l1norm;
    else
      for (int b : sig.betti) std::cout << ',' << b;
    if (sig.has_delta)
      std::cout << ',' << sig.delta;
    else
      std::cout << ',';
    if (sig.has_z)
      std::cout << ',' << sig.zscore << ',' << (sig.spike ? 1 : 0);
    else
      std::cout << ",,0";
    std::cout << std::endl;  // flush per row: required for live piped feeds
    ++rows;
    if (sig.spike) ++spikes;
  }

  std::cerr << "[info] ticks=" << ticks << " windows=" << rows << " spikes=" << spikes
            << " (w=" << cfg.w << " dim=" << det.config().dim
            << (basket ? " basket" : "") << (det.config().landscape ? " landscape" : "");
  if (!basket) std::cerr << " m=" << cfg.m << " d=" << cfg.d;
  if (!det.config().eps.empty())
    std::cerr << " eps_max=" << det.config().eps.back();
  std::cerr << ")\n";

  if (benchmark) {
    std::cerr << "[bench] per-tick latency over " << inc_stats.us.size() << " windows (w=" << cfg.w
              << ", " << cfg.w * (cfg.w - 1) / 2 << " edges)\n";
    const double im = inc_stats.mean(), nm = naive_stats.mean();
    std::cerr << "[bench]   incremental: mean=" << im << "us p50=" << inc_stats.pct(0.5)
              << "us p99=" << inc_stats.pct(0.99) << "us\n";
    std::cerr << "[bench]   naive:       mean=" << nm << "us p50=" << naive_stats.pct(0.5)
              << "us p99=" << naive_stats.pct(0.99) << "us\n";
    std::cerr << "[bench]   speedup (mean): " << (im > 0 ? nm / im : 0.0) << "x\n";
    std::cerr << "[bench]   curve mismatches vs naive: " << naive_mismatches << "\n";
    if (naive_mismatches != 0) return 1;
  }
  return 0;
}
