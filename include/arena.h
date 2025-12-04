#include "knot.h"
#include "node.h"
#include "visit.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <utility>

constexpr int64_t BATCH_SIZE = 16;
constexpr int64_t TOTAL_CROSSINGS = MAX_CROSSINGS * BATCH_SIZE;

constexpr int64_t GRAPH_DEGREE = 4;
constexpr int64_t N_FACES = 4;
constexpr int64_t N_OTHER_FEATURES = 2;

constexpr int64_t SIGN_FEATURE_IDX = 0;
constexpr int64_t VISITS_UNTIL_SELF_IDX = 1;

struct SharedArena {
  std::atomic<int64_t> cur_entry{0};
  std::atomic<int64_t> cur_vertex{0};

  pthread_mutex_t mutex;
  pthread_cond_t cond_read_ready;
  pthread_cond_t cond_write_ready;

  // inputs
  uint16_t crossing_counts[BATCH_SIZE];
  uint16_t graph[TOTAL_CROSSINGS][GRAPH_DEGREE];
  uint16_t facial_lengths[TOTAL_CROSSINGS][N_FACES];
  uint16_t other_features[TOTAL_CROSSINGS][N_OTHER_FEATURES];

  // outputs
  float probs[TOTAL_CROSSINGS][N_MOVES];
  float values[BATCH_SIZE];

  std::pair<int64_t, int64_t> load(const Knot &knot) {
    int64_t entry = cur_entry.fetch_add(1, std::memory_order_acq_rel);
    int64_t first_vertex = cur_vertex.fetch_add(knot.n_crossings, std::memory_order_acq_rel);

    if (entry >= BATCH_SIZE || first_vertex + knot.n_crossings > TOTAL_CROSSINGS) {
      std::cerr << "Arena overflow" << std::endl;
      exit(1);
    }

    std::fill_n(&facial_lengths[first_vertex][0], knot.n_crossings * N_FACES, uint16_t{0});

    knot.compute_facial_lengths(&facial_lengths[first_vertex]);
    crossing_counts[entry] = knot.n_crossings;
    knot.to_graph(&graph[first_vertex]);
    for (int64_t v = 0; v < knot.n_crossings; v++) {
      other_features[first_vertex + v][SIGN_FEATURE_IDX] = static_cast<uint16_t>(knot.vertex_sign(v));
      other_features[first_vertex + v][VISITS_UNTIL_SELF_IDX] = static_cast<uint16_t>(knot.visits_until_self(v));
    }

    if (entry == BATCH_SIZE - 1) {
      //WE HAVE TO DO SOMETHING HERE!!!
    }
    
    return {entry, first_vertex};
  }

  float unload(Node &node, int64_t first_vertex, int64_t entry) {
    double total_prob = 0.0;

    for (Child &child : node.children) {
      const uint16_t bit = Visit::MOVE_TO_BIT(child.move.move);
      const uint16_t v = child.move.v;
      child.p = std::exp(probs[first_vertex + v][bit]);
      total_prob += child.p;
    }

    const double inv = 1.0 / total_prob;

    for (Child &child : node.children) {
      child.p *= inv;
    }

    return values[entry];
  }
};

double Node::expand(SharedArena &arena) {
  if (is_expanded) {
    std::cerr << "Node already expanded" << std::endl;
    return 0.0;
  }

  is_expanded = true;

  const auto [entry, first_vertex] = arena.load(knot);

  // FIXME: WAIT

  for (auto m : knot.moves()) {
    children.emplace_back(this, m, 0.0);
  }

  return arena.unload(*this, first_vertex, entry);
}
