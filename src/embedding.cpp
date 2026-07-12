#include <torch/torch.h>


class embedding : public torch::nn::Module {
private:
    torch::nn::Embedding embedding_model {nullptr};
public:
    embedding(){

        embedding_model = register_module(
            "embedding_model",
            torch::nn::Embedding(
                torch::nn::EmbeddingOptions(
                    100, 4
                )
            )

        );
        
    }

    torch::Tensor forward(torch::Tensor input_tokens){
        return embedding_model->forward(input_tokens);
    }
};

