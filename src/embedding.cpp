#include <memory>
#include <torch/torch.h>
#include <transformerConfig.h>
#include <positionalEncoder.h>
#include <embedding.h>



EmbeddingImpl :: EmbeddingImpl(const std::shared_ptr<TransformerConfig>& config){

    DEBUG("INIT EMEBEDDINGS");
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


torch::Tensor EmbeddingImpl :: forward(torch::Tensor tokens){

    // objective 
    // should take a sequence or a single token and returns their embeddings

    DEBUG("EMBEDDING FORWARD PASS");
    DEBUG("EMBEDDING INPUT TOKENS");
    DEBUG(tokens);
    torch::Tensor embeddings = embedding_model->forward(tokens);
    DEBUG("EMBEDDINGS");
    DEBUG(embeddings);

    return embeddings;
}

// 1 x 1 -> embedding -> 1 x 1 x 4 
// {
//     {a, b, c, d}
// }




