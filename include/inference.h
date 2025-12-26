#pragma once

#include <iostream>
#include <cstdint>
#include <functional>
#include <pthread.h>
#include <stdexcept>
#include <string>

#include "arena.h"
#include "c10/core/DeviceType.h"
#include <torch/script.h>
#include <torch/torch.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
inline constexpr torch::DeviceType kInferDevice = torch::kCUDA;
#else
using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;
static constexpr unsigned int cudaHostRegisterMapped = 0;
inline const char *cudaGetErrorString(cudaError_t) { return "CUDA disabled"; }
inline cudaError_t cudaHostRegister(void *, std::size_t, unsigned int) {return cudaSuccess;}
inline cudaError_t cudaHostGetDevicePointer(void **device_ptr, void *host_ptr, unsigned int) {
  *device_ptr = host_ptr;
  return cudaSuccess;
}
inline constexpr torch::DeviceType kInferDevice = torch::kCPU;
#endif

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

    d_output_tensor = mapPinnedDevicePtr<float>(arena->logits);
    d_value_outputs = mapPinnedDevicePtr<float>(arena->values);
  }

  void run_inference(torch::jit::script::Module &model) {
    pthread_mutex_lock(&arena->mutex);
    const int64_t total_verts = arena->cur_vertex;
    const int64_t total_entries = arena->cur_entry;
    pthread_mutex_unlock(&arena->mutex);

    if (total_entries <= 0 || total_verts <= 0) {
      return;
    }

    c10::InferenceMode guard;
    model.eval();

    auto float_opts = torch::TensorOptions().device(kInferDevice).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(kInferDevice).dtype(torch::kInt16);

    auto counts_gpu_view =  torch::from_blob(d_crossing_counts, {total_entries}, int_opts);

    auto graph_gpu_view =  torch::from_blob(d_graph, {total_verts, GRAPH_DEGREE}, int_opts);
    auto other_features_gpu_view = torch::from_blob(d_other_features, {total_verts, N_OTHER_FEATURES}, int_opts);
    auto facial_lengths_gpu_view = torch::from_blob(d_facial_lengths, {total_verts, N_FACES}, int_opts);

    auto output_tuple = model.forward({facial_lengths_gpu_view, other_features_gpu_view, graph_gpu_view, counts_gpu_view})  .toTuple();

    torch::Tensor policy = output_tuple->elements()[0].toTensor(); // [total_verts, N_MOVES]
    torch::Tensor values = output_tuple->elements()[1].toTensor(); // [total_entries]

    auto output_arena_view = torch::from_blob(d_output_tensor, {total_verts, N_MOVES}, float_opts);

    output_arena_view.copy_(policy);

    auto value_arena_view = torch::from_blob(d_value_outputs, {total_entries}, float_opts);
    value_arena_view.copy_(values);

    #ifdef USE_CUDA
    cudaDeviceSynchronize();
    #endif

    pthread_mutex_lock(&arena->mutex);
    arena->entries_left = total_entries; // NOT blindly BATCH_SIZE
    pthread_cond_broadcast(&arena->cond_read_ready);
    pthread_mutex_unlock(&arena->mutex);
  }
};

double Node::expand(SharedArena &arena) {
  if (is_expanded) {
    std::cerr << "Node already expanded" << std::endl;
    return 0.0;
  }

  is_expanded = true;

  const auto [entry, first_vertex] = arena.load(knot);

  for (auto m : knot.moves()) {
    children.emplace_back(Child(this, m, 0.0));
  }

  return arena.unload(*this, entry, first_vertex);
}
