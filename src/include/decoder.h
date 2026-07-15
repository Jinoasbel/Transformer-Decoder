#pragma once
#include <transformerConfig.h>
#include <memory>
#include <torch/torch.h>
#include <feedForwardNN.h>



class DecoderImpl : public torch::nn::Module {
private:
    Embedding embedding_model {nullptr};
    Attention attention_block {nullptr};
    torch::nn::LayerNorm normalization_block {nullptr};
    FeedForwardNeuralNetwork ffnn {nullptr};

public:
    DecoderImpl(const std::shared_ptr<TransformerConfig>& config);

    torch::Tensor forward(const torch::Tensor& input_tensor);
};
TORCH_MODULE(Decoder);