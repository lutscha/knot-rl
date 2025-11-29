#include "../include/knot.h"
#include <chrono>
#include <queue>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;

struct KnotCmp {
  bool operator()(const Knot &a, const Knot &b) const noexcept {
    if (a.n_crossings != b.n_crossings) [[likely]]
      return a.n_crossings > b.n_crossings;

    if (a.hash_ != b.hash_) [[likely]]
      return a.hash_ > b.hash_;

    return false;
  }
};

struct GreedyResult {
  Knot best;
  std::vector<AvailableMove> path; // moves from start to best
};

GreedyResult greedy_minimize_crossings(Knot start, std::size_t max_expansions) {
  using PQ = std::priority_queue<Knot, std::vector<Knot>, KnotCmp>;
  using Hash = uint64_t;

  struct ParentInfo {
    Hash parent;
    AvailableMove move; // move that produced this state from parent
  };

  // parent_map presence == “visited”
  std::unordered_map<Hash, ParentInfo> parent_map;
  parent_map.reserve(max_expansions);
  parent_map.max_load_factor(0.7f);

  const Hash start_hash = start.hash_;
  // root: parent = itself, move is dummy and never read
  parent_map.emplace(
      start_hash,
      ParentInfo{start_hash, AvailableMove(0, ReidemeisterMove::R1_neg())});

  // Pre-reserve underlying storage for the heap
  std::vector<Knot> heap_storage;
  heap_storage.reserve(max_expansions + 1);
  PQ pq{KnotCmp{}, std::move(heap_storage)};
  pq.push(start);

  Knot best = start;
  Hash best_hash = start_hash;
  std::size_t expansions = 0;

  while (!pq.empty() && expansions < max_expansions) {
    Knot cur = pq.top();
    pq.pop();
    const Hash cur_hash = cur.hash_;

    if (cur.n_crossings < best.n_crossings) {
      best = cur;
      best_hash = cur_hash;
    };

    ++expansions;

    // 1) Direct moves from the iterator
    for (auto m : cur.moves()) {
      Knot child = cur.apply_move(m.v, m.move);
      const Hash h = child.hash_;

      auto [it, inserted] = parent_map.emplace(h, ParentInfo{cur_hash, m});
      if (!inserted)
        continue; // already visited
      pq.push(std::move(child));
    }

    // 2) Local R2+ moves (8 per vertex)
    const uint16_t n = cur.n_crossings;
    for (uint16_t v = 0; v < n; ++v) {
      for (VisitType type : {VisitType::over, VisitType::under}) {
        for (Direction dir_over : {Direction::next, Direction::prev}) {
          for (Direction dir_under : {Direction::next, Direction::prev}) {

            ReidemeisterMove mv = ReidemeisterMove::R2_pos(type, dir_over, dir_under);

            Knot child = cur.apply_move(v, mv);

            const Hash h = child.hash_;

            auto [it, inserted] = parent_map.emplace(
                h, ParentInfo{cur_hash, AvailableMove(v, mv)});
            if (!inserted)
              continue; // already visited
            pq.push(std::move(child));
          }
        }
      }
    }
  }

  // Reconstruct path from start_hash to best_hash.
  std::vector<AvailableMove> path;
  path.reserve(best.n_crossings + 8); // rough guess; better than nothing


    Hash h = best_hash;
  while (h != start_hash) {
    auto it = parent_map.find(h);
    if (it == parent_map.end())
      break; // should not happen; abort reconstruction

    const ParentInfo &info = it->second;
    path.push_back(info.move);
    h = info.parent;
  }
  std::reverse(path.begin(), path.end());

  return GreedyResult{std::move(best), std::move(path)};
}
