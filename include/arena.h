#include "knot.h"
#include "node.h"
#include "visit.h"
#include <atomic>
#include <cstdint>
#include <pthread.h>

constexpr int64_t BATCH_SIZE = 16;
constexpr int64_t TOTAL_CROSSINGS = MAX_CROSSINGS * BATCH_SIZE;

constexpr int64_t GRAPH_DEGREE = 4;
constexpr int64_t N_FACES = 4;
constexpr int64_t N_OTHER_FEATURES = 2;

constexpr int64_t SIGN_FEATURE_IDX = 0;
constexpr int64_t VISITS_UNTIL_SELF_IDX = 1;

void load_arena(SharedArena &arena, const Knot &knot);
float unload_arena(const SharedArena &arena, const Knot &knot, int64_t first_row, Node &node);

double Node::expand(SharedArena &arena) {
  if (is_expanded) {
    std::cerr << "Node already expanded" << std::endl;
    return 0.0;
  }

  is_expanded = true;

  load_arena(arena, knot);

  // FIXME: WAIT

  for (auto m : knot.moves()) {
    children.push_back(Child(this, m, 0.0));
  }

  uint64_t first_row = 0; // FIXME: GET FIRST ROW
  return unload_arena(arena, knot, first_row, *this);
}

struct SharedArena {
  std::atomic<int64_t> cur_row{0};

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
};

void load_arena(SharedArena &arena, const Knot &knot) {
  int64_t row = arena.cur_row.load(std::memory_order_acquire);
  knot.to_graph(&arena.graph[row]);
  for (int64_t v = 0; v < knot.n_crossings; v++) {
    arena.other_features[row + v][SIGN_FEATURE_IDX] = static_cast<uint16_t>(knot.vertex_sign(v));
    arena.other_features[row + v][VISITS_UNTIL_SELF_IDX] = static_cast<uint16_t>(knot.visits_until_self(v));
  }

  arena.cur_row.fetch_add(knot.n_crossings, std::memory_order_acquire);
}

float unload_arena(const SharedArena &arena, const Knot &knot, int64_t first_row, Node &node) {
  double total_prob = 0.0;
  for (int64_t v = 0; v < knot.n_crossings; v++) {
    const unsigned moves = static_cast<unsigned>(knot.move_mask(v));
    uint16_t bit = std::countr_zero(moves);

    while (moves >> bit){
        total_prob += std::exp(arena.probs[first_row + v][bit]);
        bit += 1 + std::countr_zero(moves >> (bit + 1));
    }
  }

  for (Child &child : node.children) {
    const uint16_t bit = Visit::MOVE_TO_BIT(child.move.move);
    const uint16_t v = child.move.v;
    total_prob += std::exp(arena.probs[first_row + v][bit]);
  }

  for (Child &child : node.children) {
     const uint16_t bit = Visit::MOVE_TO_BIT(child.move.move);
     const uint16_t v = child.move.v;
     child.p = std::exp(arena.probs[first_row + v][bit]) / total_prob;
  }

  return arena.values[first_row];
}

