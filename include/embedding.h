#pragma once

#include <torch/torch.h>
#include <memory>
#include <positionalEncoder.h>
#include <transformerConfig.h>

/**
 * @brief Wrapper around the ``torch::embedding``
 * @returns ``torch::Tensor`` of Embeddings
 * 
 * ### Example:
 * ```
 * cpp
 * auto config = std::make_shared<TransformerConfig>();
 * config->embedding_dim = 2;
 * config->no_of_embeddings = 100;
 * Embedding embedding_model(config);
 *
 * std::cout<<embedding_model->forward(torch::tensor({1,2}));
 *
 * // 0.53 4.12
 * // 5.21 2.14
 * ```
 *
 * @test no problems were found
 * @author Jino Asbel
 * @date 20-08-2026
 */
class EmbeddingImpl : public torch::nn::Module{
private:
    torch::nn::Embedding embedding_model {nullptr};

public:

    /**
     * @brief Initialize embedding model
     * @param ``std::shared_ptr<TransformerConfig>``
     * @pre ``TransformerConfig->no_of_embeddings, TransformerConfig->embedding_dim``
     * 

     */
    EmbeddingImpl(const std::shared_ptr<TransformerConfig>& config);

    /**
     * @param ``torch::Tensor`` Tokens of dimension d
     * @returns ``torch::Tensor`` embeddings of dimension d+1
     * @warning positional encodings should be added later
     * 
     */
    torch::Tensor forward(torch::Tensor tokens);
    
};

TORCH_MODULE(Embedding);

