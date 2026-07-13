#include <attention.h>
#include <embedding.h>
#include <transformer-config.h>
#include <torch/torch.h>

int main(){

    std::shared_ptr<TransformerConfig> config = std::make_shared<TransformerConfig>();
    config->embedding_dim = 4;
    config->no_of_embeddings = 100;
    config->wk_input_dim = 4;
    config->wk_output_dim = 4;
    config->wq_input_dim = 4;
    config->wq_output_dim = 4;
    config->wv_input_dim = 4;
    config->wv_output_dim = 4;

    embedding embedding_model = embedding(config);
    attention attention_block = attention(config);

    attention_block.forward(
        embedding_model.forward(
            torch::tensor({
                {1,2,7,4,9}
            })
        )
    );
    return 0;
}