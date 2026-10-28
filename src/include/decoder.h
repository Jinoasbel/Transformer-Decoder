#pragma once
#include <transformerConfig.h>
#include <memory>
#include <torch/torch.h>
#include <feedForwardNN.h>
#include <attention.h>
#include <embedding.h>


/**
 * @brief Decoder implements ``Attention``, ``LayerNorm`` and ``feedForwardNN``. Passes input tesnor to attention block then performs residual add normalization and then over a feed forward nn and finally add the output.
 * @param ``torch::Tensor`` of dim 3
 * @returns ``torch::Tensor`` of dim 3
 * @test on 04-09-2026 and found the decoder block is working fine
 */
class DecoderImpl : public torch::nn::Module {
private:
    Attention attention_block {nullptr};
    torch::nn::LayerNorm normalization_block {nullptr};
    FeedForwardNeuralNetwork ffnn {nullptr};

public:
    DecoderImpl(const std::shared_ptr<TransformerConfig>& config);

    torch::Tensor forward(const torch::Tensor& input_tensor);
};
TORCH_MODULE(Decoder);