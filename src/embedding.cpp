#include <memory>
#include <torch/torch.h>
#include <transformerConfig.h>
#include <embedding.h>


EmbeddingImpl :: EmbeddingImpl(const std::shared_ptr<TransformerConfig>& config){

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
torch::Tensor EmbeddingImpl :: forward(torch::Tensor input_tokens){

    // need to write positional encoder
    return embedding_model->forward(input_tokens);
}




