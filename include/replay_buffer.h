#pragma once

#include <vector>
#include <atomic>
#include <shared_mutex>
#include <random>
#include <cstring>
#include <tuple>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include "arena.h" // Imports constants: GRAPH_DEGREE, N_FACES, N_MOVES, etc.

constexpr size_t REPLAY_CAPACITY = 100000;       // Max games in cold storage
constexpr size_t TRAIN_BATCH_SIZE = 512;         // Games per training batch
constexpr size_t MAX_NODES_PER_BATCH = 200000;   // Pre-allocated staging space

// cold storage in heap memory that is later written to staging arena
struct StoredGame {
    std::vector<int16_t> graph;           // [N, GRAPH_DEGREE]
    std::vector<int16_t> facial_lengths;  // [N, N_FACES]
    std::vector<int16_t> other_features;  // [N, N_OTHER_FEATURES]
    
    std::vector<int16_t>   target_visits;   // [N, N_MOVES]
    float target_value;
    
    int16_t num_nodes;
};

struct ReplayArena {
    int16_t crossing_counts[TRAIN_BATCH_SIZE];
    int16_t graph[MAX_NODES_PER_BATCH][GRAPH_DEGREE];
    int16_t facial_lengths[MAX_NODES_PER_BATCH][N_FACES];
    int16_t other_features[MAX_NODES_PER_BATCH][N_OTHER_FEATURES];

    int16_t target_visits[MAX_NODES_PER_BATCH][N_MOVES];
    float target_value[TRAIN_BATCH_SIZE];
};


class ReplayBuffer {
private:
    std::vector<StoredGame> buffer;
    size_t head = 0;
    bool is_full = false;

    ReplayArena* arena;

    int16_t* d_crossing_counts = nullptr;
    int16_t* d_graph = nullptr;
    int16_t* d_facial_lengths = nullptr;
    int16_t* d_other_features = nullptr;
    int16_t* d_target_visits = nullptr;
    float* d_target_value = nullptr;

    mutable std::shared_mutex rw_lock; // rw lock thing
    std::mt19937 rng; // rng for random sampling

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
    ReplayBuffer() : rng(std::random_device{}()) {
        buffer.resize(REPLAY_CAPACITY);

        arena = new ReplayArena();

        auto cerr = cudaHostRegister(arena, sizeof(ReplayArena), cudaHostRegisterMapped);

        if (cerr != cudaSuccess) [[unlikely]] {
            delete arena;
            throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                                     cudaGetErrorString(cerr));
        }

        d_crossing_counts = mapPinnedDevicePtr<int16_t>(arena->crossing_counts);
        d_graph           = mapPinnedDevicePtr<int16_t>(arena->graph);
        d_facial_lengths  = mapPinnedDevicePtr<int16_t>(arena->facial_lengths);
        d_other_features  = mapPinnedDevicePtr<int16_t>(arena->other_features);
        d_target_visits   = mapPinnedDevicePtr<int16_t>(arena->target_visits);
        d_target_value    = mapPinnedDevicePtr<float>(arena->target_value);
    }

    ~ReplayBuffer() {
        if (arena) {
            cudaHostUnregister(arena);
            delete arena;
        }
    }

    // this is called by mcts workers probably
    void AddExample(const std::vector<int16_t>& graph,
                    const std::vector<int16_t>& facial,
                    const std::vector<int16_t>& other,
                    const std::vector<int16_t>& visits,
                    float value,
                    int16_t num_nodes) 
    {
        std::unique_lock<std::shared_mutex> lock(rw_lock);

        StoredGame& slot = buffer[head];
        
        // reuse vector capacity (avoids malloc)
        slot.graph = graph;
        slot.facial_lengths = facial;
        slot.other_features = other;
        slot.target_visits = visits;
        slot.target_value = value;
        slot.num_nodes = num_nodes;

        head = (head + 1) % REPLAY_CAPACITY;
        if (head == 0) is_full = true;
    }

    // this is called by the trainer module
    // returns tuple for the staging arena batch moved to gpu
    // does this actually work
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    Sample(int batch_size = TRAIN_BATCH_SIZE) {
        std::shared_lock<std::shared_mutex> lock(rw_lock);

        size_t max_valid = is_full ? REPLAY_CAPACITY : head;
        if (max_valid == 0) return {}; 

        int total_nodes = 0;
        int valid_batch_size = 0;

        std::uniform_int_distribution<size_t> dist(0, max_valid - 1);

        // CPU Gather Loop (Cold -> Pinned Staging)
        for (int i = 0; i < batch_size; ++i) {
            const StoredGame& game = buffer[dist(rng)];

            // overflow check
            if (total_nodes + game.num_nodes > MAX_NODES_PER_BATCH) break;

            // Memcpy is safe and fast for flat POD arrays
            // pointer arithmetic handles

            // bunch of memcopy calls
            std::memcpy(arena->graph[total_nodes], 
                        game.graph.data(), game.num_nodes * GRAPH_DEGREE * sizeof(int16_t));
            
            std::memcpy(arena->facial_lengths[total_nodes], 
                        game.facial_lengths.data(), game.num_nodes * N_FACES * sizeof(int16_t));

            std::memcpy(arena->other_features[total_nodes], 
                        game.other_features.data(), game.num_nodes * N_OTHER_FEATURES * sizeof(int16_t));

            std::memcpy(arena->target_visits[total_nodes], 
                        game.target_visits.data(), game.num_nodes * N_MOVES * sizeof(int16_t));

            arena->crossing_counts[valid_batch_size] = game.num_nodes;
            arena->target_value[valid_batch_size] = game.target_value;

            total_nodes += game.num_nodes;
            valid_batch_size++;
        }
        
        // this is just a bunch of from_blob reads to memory        
        auto int_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt16);
        auto float_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

        auto t_graph   = torch::from_blob(d_graph, {total_nodes, GRAPH_DEGREE}, int_opts);
        auto t_facial  = torch::from_blob(d_facial_lengths, {total_nodes, N_FACES}, int_opts);
        auto t_other   = torch::from_blob(d_other_features, {total_nodes, N_OTHER_FEATURES}, int_opts);
        auto t_counts  = torch::from_blob(d_crossing_counts, {valid_batch_size}, int_opts);
        
        auto t_target_visit = torch::from_blob(d_target_visits, {total_nodes, N_MOVES}, int_opts);
        auto t_target_v  = torch::from_blob(d_target_value, {valid_batch_size}, float_opts);

        return std::make_tuple(t_graph, t_facial, t_other, t_counts, t_target_visit, t_target_v);
    }
};