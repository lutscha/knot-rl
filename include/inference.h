#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>

struct InferenceResult {
    torch::Tensor policy;
    torch::Tensor value;
};

struct InferenceRequest {
    torch::Tensor features; // (N_i, 11)
    std::promise<InferenceResult> promise;
};

class InferenceEngine {
private:
    torch::jit::script::Module model;
    std::atomic<bool> running{true};
    std::thread batcher_thread;

    // --- Queue System ---
    std::vector<InferenceRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;

    // --- Optimization Config ---
    const int MAX_BATCH_GRAPHS = 256;      // Trigger if we have this many requests
    const int MAX_BUFFER_NODES = 10000;    // Pre-allocated buffer size (tuning required)
    const int LATENCY_LIMIT_US = 500;      // 0.5ms max wait time
    
    // --- The "Staging Area" (Pinned Memory) ---
    torch::Tensor pinned_input_buffer; 

    void batching_loop();

public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine();

    InferenceResult predict(const torch::Tensor& features);
};