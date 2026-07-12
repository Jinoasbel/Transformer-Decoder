#include <torch/torch.h>

class embedding : public torch::nn::Module{
private:
    torch::nn::Embedding embedding_model {nullptr};


public:
    embedding();  
    // load embedding model if exists else create a new embedding model

    torch::Tensor forward(torch::Tensor input_tokens);  
    // @args token_id's : Tensor @returns embeddings : Tensor
    
};