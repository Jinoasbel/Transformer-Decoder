#include <attention.h>
#include <torch/torch.h>
#include <transformerConfig.h>


AttentionImpl :: AttentionImpl(const std::shared_ptr<TransformerConfig>& config){

    
    wq = register_module(
        "wQ", 
        torch::nn::Linear(config->wq_input_dim, config->wq_output_dim)
    );
    wk = register_module(
        "wK",
        torch::nn::Linear(config->wk_input_dim, config->wk_output_dim)
    );
    wv = register_module(
        "wV",
        torch::nn::Linear(config->wv_input_dim, config->wv_output_dim)
    );

    dimension = config->embedding_dim;

    DEBUG("ATTENTION INITIATED");

}

// embedding : Tensor->shape [no_of_batch, batch_size, embedding_dim]
// ie 
// 2x 2x4
// 4x2 so 2x4 x 4x2
torch::Tensor AttentionImpl :: forward(const torch::Tensor& embeddings){

    DEBUG("ATTENTION FORWARD PASS");

    // preserve shape
    torch::Tensor query = wq->forward(embeddings);
    DEBUG("QUERY");
    DEBUG(query);
    torch::Tensor key = wk->forward(embeddings);
    DEBUG("KEY");
    DEBUG(key);
    torch::Tensor value = wv->forward(embeddings);
    DEBUG("VALUE");    
    DEBUG(value);

    torch::Tensor scores = torch::matmul(
        query, key.transpose(1,2)
    );
    scores = scores / std::sqrt(dimension);
    DEBUG("SCORES");
    DEBUG(scores);

    // upper triangular
    DEBUG("MASK"); 
    torch::Tensor mask = torch::triu(
        // creates a tensor and fills with the same value
        // @args shape, value 
        torch::full({scores.sizes()[1], scores.sizes()[2]}, -INFINITY),
        1
    );
    DEBUG(mask);
    // [
    //     [0 inf inf]
    //     [0  0  inf]
    //     [0  0   0 ]
    // ]    

    scores += mask;
    DEBUG("SCORES + MASK");
    DEBUG(scores + mask);
    // dimension to convert the values to probablilities -1 -> last dimension
    torch::Tensor attention = torch::softmax(scores, -1);
    DEBUG("ATTENTION");
    DEBUG(attention);

    DEBUG("ATTENTION X VALUE");
    // attention will be added at the decoder block
    DEBUG(torch::matmul(attention, value));
    return torch::matmul(attention, value);
}

torch::Tensor AttentionImpl :: forward_kv_cache(const torch::Tensor& embedding){

    DEBUG("ATTENTION FORWARD PASS (KV ENABLED)");
    
    if(embedding.size(-2) > 1){
        DEBUG("INPUT DIM INCOMPATIBLE");
        throw std::runtime_error("incompatibe embedding input in forward");
    }

    torch::Tensor query = wq->forward(embedding);
    DEBUG("QUERY");
    DEBUG(query);
    torch::Tensor key = wk->forward(embedding);
    DEBUG("KEY");
    DEBUG(key);
    k_cache = torch::cat({k_cache, key}, 0);
    DEBUG("K CACHE");
    DEBUG(k_cache);
    torch::Tensor value = wv->forward(embedding);
    DEBUG("VALUE");    
    DEBUG(value);
    v_cache = torch::cat({v_cache, value}, 0);
    DEBUG("V CACHE");
    DEBUG(v_cache);

    torch::Tensor scores = torch::matmul(
        query, k_cache.transpose(1, 2)
    );

    scores = scores / std::sqrt(dimension);
    DEBUG("SCORES");
    DEBUG(scores);

    torch::Tensor attention = torch::softmax(scores, -1);
    DEBUG("ATTENTION");
    DEBUG(attention);

    DEBUG("ATTENTION X VALUE");
    // attention will be added at the decoder block
    DEBUG(torch::matmul(attention, value));
    return torch::matmul(attention, v_cache);

}
