#include "knot.h"
#include "node.h"
#include "visit.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <tuple>
#include <algorithm>
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

  std::atomic<int64_t> entries_ready{0};
  std::atomic<int64_t> entries_left{0};

  pthread_mutex_t mutex;
  pthread_cond_t cond_read_ready;
  pthread_cond_t cond_write_ready;

  // inputs
  uint16_t crossing_counts[BATCH_SIZE];
  uint16_t graph[TOTAL_CROSSINGS][GRAPH_DEGREE];
  uint16_t facial_lengths[TOTAL_CROSSINGS][N_FACES];
  uint16_t other_features[TOTAL_CROSSINGS][N_OTHER_FEATURES];

  // outputs
  float logits[TOTAL_CROSSINGS][N_MOVES];
  float values[BATCH_SIZE];

  std::tuple<int64_t, int64_t, bool> load(const Knot &knot) {

    int64_t entry = cur_entry.fetch_add(1, std::memory_order_acq_rel);
    int64_t first_vertex = cur_vertex.fetch_add(knot.n_crossings, std::memory_order_acq_rel);

    if (entry >= BATCH_SIZE ||  first_vertex + knot.n_crossings > TOTAL_CROSSINGS) {
      std::cerr << "Arena overflow" << std::endl;
      exit(1);
    }

    std::fill_n(&facial_lengths[first_vertex][0], knot.n_crossings * N_FACES, uint16_t{0});

    crossing_counts[entry] = knot.n_crossings;
    knot.compute_facial_lengths(&facial_lengths[first_vertex]);
    knot.to_graph(&graph[first_vertex]);
    for (int64_t v = 0; v < knot.n_crossings; v++) {
      other_features[first_vertex + v][SIGN_FEATURE_IDX] = static_cast<uint16_t>(knot.vertex_sign(v));
      other_features[first_vertex + v][VISITS_UNTIL_SELF_IDX] =  static_cast<uint16_t>(knot.visits_until_self(v));
    }

    const auto result = std::make_tuple(entry, first_vertex, entry == BATCH_SIZE - 1);

    const uint64_t entries_ready_ = entries_ready.fetch_add(1, std::memory_order_acq_rel);
    if (entries_ready_ == BATCH_SIZE - 1) {
      entries_left.store(BATCH_SIZE, std::memory_order_release);
      pthread_cond_signal(&cond_read_ready);
    }

    return result;
  }

  float unload(Node &node, int64_t entry, int64_t first_vertex) {
    double total_prob = 0.0;

    for (Child &child : node.children) {
      const uint16_t bit = Visit::MOVE_TO_BIT(child.move.move);
      const uint16_t v = child.move.v;
      child.p = std::exp(logits[first_vertex + v][bit]);
      total_prob += child.p;
    }

    if (total_prob == 0.0) {
      std::cerr << "Total probability is 0.0" << std::endl;
      exit(1);
    }

    const double inv = 1.0 / total_prob;

    for (Child &child : node.children) {
      child.p *= inv;
    }

    const float result = values[entry];

    const uint64_t entries_left_ = entries_left.fetch_add(-1, std::memory_order_acq_rel);
    if (entries_left_ == 1) {
      entries_ready.store(0, std::memory_order_release);
      cur_entry.store(0, std::memory_order_release);
      cur_vertex.store(0, std::memory_order_release);
      pthread_cond_signal(&cond_write_ready);
    }

    return result;
  }
};
