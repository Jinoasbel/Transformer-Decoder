#pragma once
#include <torch/torch.h>
#include <transformerConfig.h>

class FeedForwardNeuralNetworkImpl : public torch::nn::Module{
private:
    torch::nn::Linear layer1 {nullptr};
    torch::nn::Linear layer2 {nullptr};
public:
    FeedForwardNeuralNetworkImpl(const std::shared_ptr<TransformerConfig>& config);
    torch::Tensor forward(const torch::Tensor& normalized_input);
};
TORCH_MODULE(FeedForwardNeuralNetwork);