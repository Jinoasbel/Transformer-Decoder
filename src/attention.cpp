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

    mode = config->is_training_enabled;
    use_kv_cache = config->use_kv_cache;

    DEBUG("ATTENTION INITIATED");

}

// embedding : Tensor->shape [no_of_batch, batch_size, embedding_dim]
// ie 
// 2x 2x4
// 4x2 so 2x4 x 4x2
torch::Tensor AttentionImpl :: forward(const torch::Tensor& embeddings){


    DEBUG("ATTENTION FORWARD PASS");
    torch::Tensor scores;

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

    
    // mode : inference or training, 0 for inference 1 for training
    if(!mode){ // if not inference
        DEBUG("INFERENCE");

        if(use_kv_cache){

            // takes a batch [batch_size, 1, dimension]
            if(embeddings.size(1) == 1){
                DEBUG("INPUT DIM INCOMPATIBLE");
                throw std::runtime_error("incompatibe embedding input in forward");
            }
            // if kv cache is enabled 
            DEBUG("KV CACHE ENABLED");
            if(!k_cache.defined()){
                k_cache = key;
                v_cache = value;
            }
            else {
                k_cache = torch::cat({k_cache, key}, 0);
                v_cache = torch::cat({v_cache, value}, 0);
            }

            DEBUG("NEW KV CACHE");
            DEBUG(k_cache);
            DEBUG(v_cache);

            scores = torch::matmul(
                query, k_cache.transpose(1 , 2)
            ) / std::sqrt(embeddings.size(-1));

        }

        else {    
            DEBUG("KV CACHE DISABLED");
            scores = torch::matmul(
                query, key.transpose(1 , 2)
            ) / std::sqrt(embeddings.size(-1));
        }
                
    }
    else {

        scores = torch::matmul(
            query, key.transpose(1 , 2)
        ) / std::sqrt(embeddings.size(-1));

        DEBUG("TRAINING");
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
        DEBUG("SCORES + MASK");
        scores = scores + mask;        
    }
    DEBUG(scores);
    // dimension to convert the values to probablilities -1 -> last dimension
    torch::Tensor attention = torch::softmax(scores, -1);
    DEBUG("ATTENTION");
    DEBUG(attention);

    DEBUG("ATTENTION X VALUE");
    // funal attention output
    DEBUG(torch::matmul(attention, value));
    return torch::matmul(attention, value);
}
