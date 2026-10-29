#pragma once

#include <torch/torch.h>
#include <transformerConfig.h>
#include <positionalEncoder.h>
#include <decoder.h>
#include <embedding.h>
#include <heads.h>
#include <vector>

class TransformerImpl : torch::nn::Module{

private:
    std::vector<Decoder> collection_of_decoders;
    Embedding emb_model {nullptr};
    Heads prediction_heads {nullptr};

    
public:
    TransformerImpl(const std::shared_ptr<TransformerConfig>& config);
    torch::Tensor forward(torch::Tensor& input, int token_no);
};
TORCH_MODULE(Transformer);