#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <pthread.h>
#include <stdexcept>
#include <string>

#include "arena.h"
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>

class InferenceServer {
public:
  SharedArena *arena;

private:
  int16_t *d_crossing_counts = nullptr;
  int16_t *d_graph = nullptr;
  int16_t *d_facial_lengths = nullptr;
  int16_t *d_other_features = nullptr;

  float *d_output_tensor = nullptr;
  float *d_value_outputs = nullptr;
  // Small helper that hides the cudaHostGetDevicePointer boilerplate.

  template <typename T> static T *mapPinnedDevicePtr(void *host_ptr) {
    void *dev_ptr = nullptr;
    auto err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (err != cudaSuccess) [[unlikely]] {
      throw std::runtime_error(
          std::string("cudaHostGetDevicePointer failed for ") +
          std::string(cudaGetErrorString(err)));
    }
    return static_cast<T *>(dev_ptr);
  }

public:
  explicit InferenceServer() : arena(new SharedArena()) {
    // initialize mutex/conds for thread sharing
    if (pthread_mutex_init(&arena->mutex, nullptr) != 0) {
      delete arena;
      throw std::runtime_error("pthread_mutex_init failed");
    }
    if (pthread_cond_init(&arena->cond_read_ready, nullptr) != 0) {
      pthread_mutex_destroy(&arena->mutex);
      delete arena;
      throw std::runtime_error("pthread_cond_init(cond_read_ready) failed");
    }
    if (pthread_cond_init(&arena->cond_write_ready, nullptr) != 0) {
      pthread_cond_destroy(&arena->cond_read_ready);
      pthread_mutex_destroy(&arena->mutex);
      delete arena;
      throw std::runtime_error("pthread_cond_init(cond_write_ready) failed");
    }

    auto cerr =
        cudaHostRegister(arena, sizeof(SharedArena), cudaHostRegisterMapped);
    if (cerr != cudaSuccess) [[unlikely]] {
      pthread_cond_destroy(&arena->cond_write_ready);
      pthread_cond_destroy(&arena->cond_read_ready);
      pthread_mutex_destroy(&arena->mutex);
      delete arena;
      throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                               cudaGetErrorString(cerr));
    }

    d_crossing_counts = mapPinnedDevicePtr<int16_t>(arena->crossing_counts);
    d_graph = mapPinnedDevicePtr<int16_t>(arena->graph);
    d_facial_lengths = mapPinnedDevicePtr<int16_t>(arena->facial_lengths);
    d_other_features = mapPinnedDevicePtr<int16_t>(arena->other_features);

    d_output_tensor = mapPinnedDevicePtr<float>(arena->probs);
    d_value_outputs = mapPinnedDevicePtr<float>(arena->values);
  }

  void run_inference(const torch::jit::script::Module &model) {
    const int64_t total_rows = arena->cur_row.load(std::memory_order_acquire);
    if (total_rows <= 0) {
      return;
    }

    auto float_opts =
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto int_opts =
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt16);

    // zero-copy views on mapped host memory (as device pointers)

    auto counts_gpu_view =
        torch::from_blob(d_crossing_counts, {BATCH_SIZE}, int_opts);

    auto graph_gpu_view =
        torch::from_blob(d_graph, {total_rows, GRAPH_DEGREE}, int_opts);

    auto other_features_gpu_view = torch::from_blob(
        d_other_features, {total_rows, N_OTHER_FEATURES}, int_opts);

    auto facial_lengths_gpu_view =
        torch::from_blob(d_facial_lengths, {total_rows, N_FACES}, int_opts);

    auto output_tuple =
        model
            .forward({facial_lengths_gpu_view, other_features_gpu_view,
                      graph_gpu_view, counts_gpu_view})
            .toTuple();

    torch::Tensor policy =
        output_tuple->elements()[0].toTensor(); // [total_rows, N_MOVES]
    torch::Tensor values =
        output_tuple->elements()[1].toTensor(); // [BATCH_SIZE]

    auto output_arena_view =
        torch::from_blob(d_output_tensor, {total_rows, N_MOVES}, float_opts);

    output_arena_view.copy_(policy);

    auto value_arena_view =
        torch::from_blob(d_value_outputs, {BATCH_SIZE}, float_opts);
    value_arena_view.copy_(values);

    cudaDeviceSynchronize();

    arena->cur_row.store(0, std::memory_order_release);
  }
};
