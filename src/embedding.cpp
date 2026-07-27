#include <memory>
#include <torch/torch.h>
#include <transformerConfig.h>
#include <positionalEncoder.h>
#include <embedding.h>



EmbeddingImpl :: EmbeddingImpl(const std::shared_ptr<TransformerConfig>& config) : use_pe_p(config->use_pre_calculated_pe){

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
    DEBUG("PERFORMANCE MODE");
    DEBUG(use_pe_p);
    if(use_pe_p){
        pe = std::make_shared<PositionalEncoder>(config);
    }
    
}

// input_tokens : torch.shape[no_of_batch > 0, batch_size]

// if input.sizes -> [a, b]
// returns [a, b, embedding dimension]

torch::Tensor EmbeddingImpl :: forward(torch::Tensor input_tokens){

    DEBUG("EMBEDDING FORWARD PASS");
    DEBUG(input_tokens);
    torch::Tensor embeddings = embedding_model->forward(input_tokens);
    DEBUG("EMBEDDINGS");
    DEBUG(embeddings);
    torch::Tensor positions = (use_pe_p)? pe->EncodePositionP(input_tokens.size(-2)) : PositionalEncoder::EncodePosition(embeddings.size(-2), embeddings.size(-1));
    DEBUG("POSITIONS");
    DEBUG(positions);

    DEBUG("EMBEDDING + POSITIONS");
    DEBUG((embeddings + positions));
    return (embeddings + positions);
}

// 1 x 1 -> embedding -> 1 x 1 x 4 
// {
//     {a, b, c, d}
// }




