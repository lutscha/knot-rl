#pragma once

#include "inference.h" // Assumes this defines InferenceServer and SharedArena
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <memory>
#include <iostream>

class Trainer {
    torch::jit::script::Module train_model; 
    torch::jit::script::Module loss_module;
    std::unique_ptr<torch::optim::Adam> optimizer;
    InferenceServer* server; 

    // Hyperparameters
    float learning_rate = 1e-3;
    float weight_decay = 1e-4;

    torch::Device device = torch::kCUDA;
    
public:
    Trainer(std::string model_path, InferenceServer* srv) : server(srv) {
        try {
            train_model = torch::jit::load(model_path);
            train_model.to(torch::kCUDA);
            train_model.train();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading training model: " << e.msg() << std::endl;
            exit(1);
        }

        try {
            loss_module = torch::jit::load("loss_module.pt");
            loss_module.to(torch::kCUDA);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading loss module: " << e.msg() << std::endl;
            exit(1);
        }

        std::vector<torch::Tensor> params;
        for (const auto& param : train_model.parameters()) {
            params.push_back(param);
        }
        
        torch::optim::AdamOptions opts(learning_rate);
        opts.weight_decay(weight_decay);
        optimizer = std::make_unique<torch::optim::Adam>(params, opts);

        train_model.to(device);
    }

    void train_step() {
        // replay buffer blackbox!!!!!!!!!!!!
        // -------------------------------
        TrainingBatch batch = Examples();
        if (batch.total_rows == 0) return;

        auto x = torch::tensor(batch.input_features, torch::dtype(torch::kInt16))
                    .view({-1, NUM_FEATURES})
                    .to(device);
        
        auto graph = torch::tensor(batch.input_graph, torch::dtype(torch::kInt16))
                        .view({-1, GRAPH_DEGREE})
                        .to(device);
        
        auto counts = torch::tensor(batch.crossing_counts, torch::dtype(torch::kInt16))
                        .to(device);        
        
        auto target_pi = torch::tensor(batch.target_probs, torch::dtype(torch::kInt16))
                            .view({-1, N_MOVES})
                            .to(device);

        auto target_v = torch::tensor(batch.target_values, torch::dtype(torch::kFloat32))
                            .to(device);

        optimizer->zero_grad();
        auto output = train_model.forward({x, graph, counts}).toTuple();
        
        torch::Tensor logits = output->elements()[0].toTensor();     
        torch::Tensor value_pred = output->elements()[1].toTensor(); 

        auto loss_inputs = std::vector<torch::jit::IValue>{
            logits, 
            value_pred, 
            target_pi, 
            target_v, 
            counts // Must be on GPU
        };
        
        torch::Tensor total_loss = loss_module.forward(loss_inputs).toTensor();

        total_loss.backward();
        optimizer->step();
    }

    void sync_model() {
        server->update_model_weights(train_model);
    }
};