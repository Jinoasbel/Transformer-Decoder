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
    torch::Tensor forward(torch::Tensor input_tokens);  
    // @args token_id's : Tensor @returns embeddings : Tensor
    
};

TORCH_MODULE(Embedding);

// note
// forward(input_tokens) : tensor.shape [block][row][column]
// {
//     {
//         {1,2,3},
//         {7,4,2},
//         {7,3,2}
//     },
//     {
//         {3,1,8},
//         {7,2,8},
//         {6,7,3}
//     }
// }
// The shape of this is [2,3,3]
// if input_token.shape = [2,3] then
// output would be [2,3,emedding dim]
// ie
// { {1,2,3}, {1,3,4} } returns
// {
//     {
//         {vec1},
//         {vec2},
//         {vec3}
//     },
//     {
//         {vec1},
//         {vec2},
//         {vec3}
//     }
// }