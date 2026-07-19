#pragma once

#include <torch/torch.h>
#include <transformerConfig.h>

class AttentionImpl : public torch::nn::Module{

private:
    torch::nn::Linear wq {nullptr};
    torch::nn::Linear wk {nullptr};
    torch::nn::Linear wv {nullptr};

    torch::Tensor mask;

public:
    AttentionImpl(const std::shared_ptr<TransformerConfig>& config); 
    // init query, key, value
    // q k v linear(embedding_dim, total_heads x embedding_dim/total_heads)
    // return shape unkown yet to be studied

    torch::Tensor forward(const torch::Tensor& embeddings);
    // @args embeddings->forward(tokens) : Tensor
    // shape to be studied
    // compute 
    // q, k, v
    // scores + masks
    // softmax(scores)
    // attention x value or v
};
TORCH_MODULE(Attention);