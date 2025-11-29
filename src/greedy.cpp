#include "../include/arena.h"
#include "../include/knot.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

  std::unordered_set<Hash> visited;
  visited.reserve(max_expansions);

  std::unordered_map<Hash, Hash> parent;
  std::unordered_map<Hash, AvailableMove> parent_move;
  parent.reserve(max_expansions);
  parent_move.reserve(max_expansions);

  const Hash start_hash = start.hash_;
  visited.insert(start_hash);

  PQ pq;
  pq.push(start);

  Knot best = start;
  Hash best_hash = start_hash;
  std::size_t expansions = 0;

  while (!pq.empty() && expansions < max_expansions) {
    Knot cur = pq.top();
    pq.pop();
    Hash cur_hash = cur.hash_;

    if (cur.n_crossings < best.n_crossings) {
      best = cur;
      best_hash = cur_hash;
    }

    ++expansions;

    // 1) Direct moves from the iterator
    for (auto m : cur.moves()) {
      Knot child = cur.apply_move(m.v, m.move);
      Hash h = child.hash_;
      if (!visited.insert(h).second)
        continue;

      parent.emplace(h, cur_hash);
      parent_move.emplace(h, m); // no default-ctor needed
      pq.push(std::move(child));
    }

    // 2) Local R2+ moves (8 per vertex)
    const uint16_t n = cur.n_crossings;
    for (uint16_t v = 0; v < n; ++v) {
      for (VisitType type : {VisitType::over, VisitType::under}) {
        for (Direction dir_over : {Direction::next, Direction::prev}) {
          for (Direction dir_under : {Direction::next, Direction::prev}) {
            ReidemeisterMove mv =
                ReidemeisterMove::R2_pos(type, dir_over, dir_under);

            Knot child = cur.apply_move(v, mv);
            Hash h = child.hash_;
            if (!visited.insert(h).second)
              continue;

            parent.emplace(h, cur_hash);
            parent_move.emplace(h, AvailableMove(v, mv));
            pq.push(std::move(child));
          }
        }
      }
    }
  }

  // Reconstruct path from start_hash to best_hash.
  std::vector<AvailableMove> path;
  Hash h = best_hash;
  while (h != start_hash) {
    auto it_p = parent.find(h);
    auto it_mv = parent_move.find(h);
    if (it_p == parent.end() || it_mv == parent_move.end()) {
      break; // inconsistent map; abort reconstruction
    }
    path.push_back(it_mv->second);
    h = it_p->second;
  }
  std::reverse(path.begin(), path.end());

  return GreedyResult{best, std::move(path)};
}
