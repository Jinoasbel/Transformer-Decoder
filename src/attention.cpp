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

}

// embedding : Tensor->shape [no_of_batch, batch_size, embedding_dim]
// ie 
// 2x 2x4
// 4x2 so 2x4 x 4x2
torch::Tensor AttentionImpl :: forward(const torch::Tensor& embeddings){

    // preserve shape
    torch::Tensor query = wq->forward(embeddings);
    torch::Tensor key = wk->forward(embeddings);
    torch::Tensor value = wv->forward(embeddings);


    torch::Tensor scores = torch::matmul(
        query, key.transpose(1,2)
    );

    // upper triangular 
    mask = torch::triu(
        // creates a tensor and fills with the same value
        // @args shape, value 
        torch::full({scores.sizes()[1], scores.sizes()[2]}, -INFINITY),
        1
    );
    // [
    //     [0 inf inf]
    //     [0  0  inf]
    //     [0  0   0 ]
    // ]    

    scores += mask;
    // dimension to convert the values to probablilities -1 -> last dimension
    torch::Tensor attention = torch::softmax(scores, -1);

    return torch::matmul(attention, value);
}
