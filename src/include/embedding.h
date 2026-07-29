#pragma once

#include <torch/torch.h>
#include <memory>
#include <positionalEncoder.h>
#include <transformerConfig.h>




class EmbeddingImpl : public torch::nn::Module{
private:
    torch::nn::Embedding embedding_model {nullptr};
    std::shared_ptr<PositionalEncoder> pe;
    bool use_pe_p;


public:
    EmbeddingImpl(const std::shared_ptr<TransformerConfig>& config);  
    // load embedding model if exists else create a new embedding model
    // input_tokens : torch.shape[no_of_batch > 0, batch_size]

    torch::Tensor forward(torch::Tensor tokens);
    // returns embeddings no positional encodings 
    // @args token_id's : Tensor @returns embeddings : Tensor
    // if args 1 x 1 returns 1 x 1 x 4
    
};

TORCH_MODULE(Embedding);

