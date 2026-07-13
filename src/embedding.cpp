#include <memory>
#include <torch/torch.h>
#include <transformer-config.h>
#include <embedding.h>


embedding :: embedding(const std::shared_ptr<TransformerConfig>& config){

    embedding_model = register_module(
        "embedding_model",
        torch::nn::Embedding(
            torch::nn::EmbeddingOptions(
                config->no_of_embeddings,
                config->embedding_dim
            )
        )

    );
    
}

// input_tokens : torch.shape[no_of_batch > 0, batch_size]
torch::Tensor embedding :: forward(torch::Tensor input_tokens){
    return embedding_model->forward(input_tokens);
}




