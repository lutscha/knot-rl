#include "../include/inference.h"
#include <chrono>
#include <cstring> // for std::memcpy

InferenceEngine::InferenceEngine(const std::string& model_path) {
    // Load Model
    try {
        model = torch::jit::load(model_path, torch::kCUDA);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.msg() << "\n";
        exit(-1);
    }

    // GPU can read from this area 2x faster than standard RAM.
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU)
        .pinned_memory(true);

    pinned_input_buffer = torch::empty({MAX_BUFFER_NODES, 11}, options);

    batcher_thread = std::thread(&InferenceEngine::batching_loop, this);
}

InferenceEngine::~InferenceEngine() {
    running = false;
    cv.notify_one();
    if(batcher_thread.joinable()) batcher_thread.join();
}

InferenceResult InferenceEngine::predict(const torch::Tensor& features) {
    std::promise<InferenceResult> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        request_queue.push_back({features, std::move(promise)});
    }
    cv.notify_one(); // wake up the driver immediately

    return future.get(); // block until result is ready
}

void InferenceEngine::batching_loop() {
    std::vector<InferenceRequest> current_batch;
    current_batch.reserve(MAX_BATCH_GRAPHS);

    while (running) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Wait logic:
            // Sleep until we have work OR we are shutting down
            cv.wait(lock, [&]{ return !request_queue.empty() || !running; });

            if (!running && request_queue.empty()) break;

            // If the queue is small, wait a tiny bit to see if more friends arrive
            // 99% of the time we will fill max_batch_size = num workers instantly
            if (request_queue.size() < MAX_BATCH_GRAPHS) {
                lock.unlock(); 
                std::this_thread::sleep_for(std::chrono::microseconds(LATENCY_LIMIT_US));
                lock.lock();
            }

            // Swap queues (Move semantics = fast)
            current_batch.swap(request_queue);
            // request_queue is now empty, ready for MCTS threads to fill again
        }

        if (current_batch.empty()) continue;

        // Instead of torch::cat, we memcpy directly into our pre-allocated pinned buffer.
        
        float* buffer_ptr = pinned_input_buffer.data_ptr<float>();
        int64_t total_nodes = 0;
        std::vector<int64_t> sizes; // to remember how to slice results back
        sizes.reserve(current_batch.size());

        for (const auto& req : current_batch) {
            int64_t n = req.features.size(0);
            
            // Safety check for buffer overflow
            if (total_nodes + n > MAX_BUFFER_NODES) {
                std::cerr << "Buffer overflow! Increase MAX_BUFFER_NODES.\n";
                // In prod, you would split the batch here. For now, we skip or panic.
                break; 
            }

            // FAST COPY: Source -> Pinned Buffer
            // 11 floats per node * n nodes * 4 bytes per float
            std::memcpy(buffer_ptr + (total_nodes * 11), 
                        req.features.data_ptr<float>(), 
                        n * 11 * sizeof(float));
            
            total_nodes += n;
            sizes.push_back(n);
        }

        // Transfer (Pinned CPU -> GPU)
        // non_blocking=true allows the CPU to immediately start cleaning up
        // while the DMA engine handles the transfer.
        
        // We take a 'slice' of the buffer (0 to total_nodes)
        auto gpu_input = pinned_input_buffer.slice(0, 0, total_nodes)
                                            .to(torch::kCUDA, /*non_blocking=*/true); // maybe get rid of non_blocking?

        // Inference (GPU)
        torch::Tensor policy, value;
        {
            torch::NoGradGuard no_grad; // Disable autograd overhead
            auto output = model.forward({gpu_input}).toTuple();
            policy = output->elements()[0].toTensor(); 
            value  = output->elements()[1].toTensor();
        }

        // Transfer Back (GPU -> CPU)
        // This is a synchronization point. We must wait for results to exist on CPU.
        policy = policy.cpu();
        value = value.cpu();

        // Distribute
        int64_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            int64_t n = sizes[i];
            
            // slicing is zero-copy (just pointer arithmetic)
            // Policy: (N_i, PolDim)
            // Value: (1, 1) - output is (Batch, 1)
            current_batch[i].promise.set_value({
                policy.slice(0, offset, offset + n),
                value[i]
            });
            
            offset += n;
        }

        // cleanup
        current_batch.clear();
    }
}