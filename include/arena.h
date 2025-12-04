#include <atomic>
#include <cstdint>
#include <pthread.h>
#include "knot.h"


constexpr int64_t MAX_CROSSINGS = 1000;
constexpr int64_t BATCH_SIZE = 16;
constexpr int64_t TOTAL_CROSSINGS = MAX_CROSSINGS * BATCH_SIZE;

constexpr int64_t GRAPH_DEGREE = 4;
constexpr int64_t N_FACES = 4;
constexpr int64_t N_OTHER_FEATURES = 2;
constexpr int64_t N_MOVES = 10;

struct SharedArena {
  std::atomic<int64_t> curr_row{0};

  pthread_mutex_t mutex;
  pthread_cond_t cond_read_ready;
  pthread_cond_t cond_write_ready;

  // inputs
  int16_t crossing_counts[BATCH_SIZE];
  int16_t graph[TOTAL_CROSSINGS][GRAPH_DEGREE];
  int16_t facial_lengths[TOTAL_CROSSINGS][N_FACES];
  int16_t other_features[TOTAL_CROSSINGS][N_OTHER_FEATURES];

  // outputs
  float output_tensor[TOTAL_CROSSINGS][N_MOVES];
  float value_outputs[BATCH_SIZE];
};

void load_arena(SharedArena &arena, const Knot &knot) {
  int64_t row = arena.curr_row.load(std::memory_order_acquire);
  knot.to_graph(arena.graph[row]);
}