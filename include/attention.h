#pragma once

#include <torch/torch.h>
#include <transformerConfig.h>


/**
 * @brief Attention Implementation of the transformer architecture, Initiates q k v(query key value) three linear layer weights are combined to improve performance, with a k v cache mechanism which works on inference and can be switched based on the needs
 */
class AttentionImpl : public torch::nn::Module{

private:
    torch::nn::Linear wq {nullptr};
    torch::nn::Linear wk {nullptr};
    torch::nn::Linear wv {nullptr};

    torch::Tensor k_cache, v_cache;

    bool mode, use_kv_cache;

public:

    /**
     * @param ``shared_ptr<TransformerConfig> config ``
     * @pre ``config->wq_input_dim, config->wq_output_dim, config->is_training_enabled, config->use_kv_cache``
     */
    AttentionImpl(const std::shared_ptr<TransformerConfig>& config); 
    

    /**
     * @brief compute query key value with or without using kv cache 
     * @pre requires embeddings to be positionally encoded
     * @warning input embedding size must be 3
     * @param ``torch::Tensor embedding``
     * @returns ``attention x value``
     * 
     */
    torch::Tensor forward(const torch::Tensor& embeddings);

};
TORCH_MODULE(Attention);