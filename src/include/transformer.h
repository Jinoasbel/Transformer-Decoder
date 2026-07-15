#pragma once

#include <torch/torch.h>
#include <transformerConfig.h>
#include <decoder.h>
#include <vector>

class TransformerImpl : torch::nn::Module{

private:
    std::vector<Decoder> collection_of_decoders;
    
public:
    TransformerImpl(const std::shared_ptr<TransformerConfig>& config);
    torch::Tensor forward(const torch::Tensor& input);
};
TORCH_MODULE(Transformer);