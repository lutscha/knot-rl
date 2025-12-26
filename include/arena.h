#pragma once

#include "knot.h"
#include "node.h"
#include "visit.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <pthread.h>
#include <tuple>

constexpr int64_t BATCH_SIZE = 16;
constexpr int64_t TOTAL_CROSSINGS = MAX_CROSSINGS * BATCH_SIZE;

constexpr int64_t GRAPH_DEGREE = 4;
constexpr int64_t N_FACES = 4;
constexpr int64_t N_OTHER_FEATURES = 2;

constexpr int64_t SIGN_FEATURE_IDX = 0;
constexpr int64_t VISITS_UNTIL_SELF_IDX = 1;

struct SharedArena {
  int64_t cur_entry = 0;
  int64_t cur_vertex = 0;

  int64_t entries_left = 0;

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

  std::tuple<int64_t, int64_t> load(const Knot &knot) {
    pthread_mutex_lock(&mutex);
    while (cur_entry >= BATCH_SIZE || cur_vertex + knot.n_crossings > TOTAL_CROSSINGS) {
      pthread_cond_wait(&cond_write_ready, &mutex);
    }

    const int64_t entry = cur_entry;
    const int64_t first_vertex = cur_vertex;
    cur_entry++;
    cur_vertex += knot.n_crossings;
    pthread_mutex_unlock(&mutex);

    if (entry >= BATCH_SIZE || first_vertex + knot.n_crossings > TOTAL_CROSSINGS) [[unlikely]] {
        throw std::runtime_error("Arena overflow");
    }

    std::fill_n(&facial_lengths[first_vertex][0], knot.n_crossings * N_FACES,uint16_t{0});

    crossing_counts[entry] = knot.n_crossings;
    knot.compute_facial_lengths(&facial_lengths[first_vertex]);
    knot.to_graph(&graph[first_vertex]);
    for (int64_t v = 0; v < knot.n_crossings; v++) {
        static_assert(N_OTHER_FEATURES == 2, "N_OTHER_FEATURES must be 2");
        other_features[first_vertex + v][SIGN_FEATURE_IDX] = static_cast<uint16_t>(knot.vertex_sign(v));
        other_features[first_vertex + v][VISITS_UNTIL_SELF_IDX] = static_cast<uint16_t>(knot.visits_until_self(v));
    }

    const auto result = std::make_tuple(entry, first_vertex);

    return result;
    }

  float unload(Node &node, int64_t entry, int64_t first_vertex) {
    pthread_mutex_lock(&mutex);
    while (entries_left == 0) {
      pthread_cond_wait(&cond_read_ready, &mutex);
    }
    pthread_mutex_unlock(&mutex);

    if (node.children.empty()) [[unlikely]] {
      throw std::runtime_error("Node has no children");
    }

    double total_prob = 0.0;
    double max_logit = -std::numeric_limits<double>::infinity();
    for (Child &child : node.children) {
      const uint16_t bit = Visit::MOVE_TO_BIT(child.move.move);
      const uint16_t v = child.move.v;
      child.p = logits[first_vertex + v][bit];
      max_logit = std::max(max_logit, child.p);
    }
    for (Child &child : node.children) {
      child.p = std::exp(child.p - max_logit);
      total_prob += child.p;
    }
    if (total_prob == 0.0) [[unlikely]] {
      throw std::runtime_error("Total probability is 0.0");
    }
    const double inv = 1.0 / total_prob;
    for (Child &child : node.children) {
      child.p *= inv;
    }

    const float result = values[entry];

    pthread_mutex_lock(&mutex);
    if (--entries_left == 0) {
      cur_entry = 0;
      cur_vertex = 0;
      pthread_cond_signal(&cond_write_ready);
    }
    pthread_mutex_unlock(&mutex);

    return result;
  }
};
