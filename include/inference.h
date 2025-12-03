#pragma once

#include <atomic>
#include <cstdint>
#include <pthread.h>
#include <stdexcept>

//#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>

constexpr uint16_t MAX_CROSSINGS = 1000; // >>16*E[num_nodes per worker]
constexpr uint16_t BATCH_SIZE = 16;      // number of workers
constexpr uint16_t TOTAL_CROSSINGS = MAX_CROSSINGS * BATCH_SIZE; // number of workers

constexpr uint16_t GRAPH_DEGREE = 4;
constexpr uint16_t NUM_FEATURES = 4 + 1 + 1;
constexpr uint16_t N_MOVES = 10;

struct SharedArena {
  // atomics here?
  std::atomic<int> curr_row;

  // Process-Shared Mutex/CondVar (Requires PTHREAD_PROCESS_SHARED attribute
  // during init)
  pthread_mutex_t mutex;
  pthread_cond_t cond_read_ready;  // Workers wait on this
  pthread_cond_t cond_write_ready; // Server waits on this

  uint16_t node_counts[BATCH_SIZE];     // how many nodes per worker, needed by model
  float value_outputs[BATCH_SIZE]; // values returned by AlphaKnot

  uint16_t input_graph[TOTAL_CROSSINGS] [GRAPH_DEGREE]; // workers write their data // here, 6
                                      // embedding + 4
  uint16_t input_features[TOTAL_CROSSINGS] [NUM_FEATURES]; // workers write their data here, 6
                                         // embedding + 4
  double output_tensor[TOTAL_CROSSINGS][N_MOVES]; // logits are stored here,
                                                  // N_MOVES moves per node
};

// called by server to run inference
// TODO: this might be suboptimal considering the model gets updated every once
// in a while

// the transfers here should be
// Worker Write: CPU Registers → Shared RAM (Zero Copy: Direct write).
// Model Input: Shared RAM → GPU VRAM (Hardware Copy: PCIe DMA).
//     - Note: Triggered transparently by the model reading the mapped memory.
// Model Exec: GPU VRAM → GPU Cores → GPU VRAM (Internal).
// Result Capture: GPU VRAM → Shared RAM (Hardware Copy: PCIe DMA via .copy_()).
// Worker Read: Shared RAM → CPU Registers (Zero Copy: Direct read).

class InferenceServer {
  SharedArena *arena;

  void *d_input_ptr;
  void *d_count_ptr;
  void *d_output_ptr;
  void *d_value_ptr;
  // list of pointers to workers for waking them up as well perhaps?

public:
  InferenceServer(SharedArena *a) : arena(a) {
    // using synchronized batching so these are fixed
    cudaHostGetDevicePointer(&d_input_ptr, arena->input_tensor, 0);
    cudaHostGetDevicePointer(&d_count_ptr, arena->node_counts, 0);
    cudaHostGetDevicePointer(&d_output_ptr, arena->output_tensor, 0);
    cudaHostGetDevicePointer(&d_value_ptr, arena->value_outputs, 0);

  }

  void run_inference(torch::jit::script::Module &model) {


    int total_rows = arena->curr_row.load();

    auto float_opts =
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto int_opts =
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

    // these are zero-copy views
    auto input_gpu_view =
        torch::from_blob(d_input_ptr, {total_rows, 10}, float_opts);
    auto counts_gpu_view = torch::from_blob(d_count_ptr, {16}, int_opts);

    auto output_tuple =
        model.forward({input_gpu_view, counts_gpu_view}).toTuple();

    torch::Tensor out_tensor_gpu = output_tuple->elements()[0].toTensor();
    torch::Tensor out_scalars_gpu = output_tuple->elements()[1].toTensor();

    // copy happens here
    auto output_arena_view =
        torch::from_blob(d_output_ptr, {total_rows, 10}, float_opts);
    output_arena_view.copy_(out_tensor_gpu);

    auto value_arena_view = torch::from_blob(d_value_ptr, {16}, float_opts);
    value_arena_view.copy_(out_scalars_gpu);

    // blocks until copy is finished for sync, is this necessary?
    cudaDeviceSynchronize();

    arena->curr_row_ptr.store(0);
    arena->active_batches.store(0);
  }
};
