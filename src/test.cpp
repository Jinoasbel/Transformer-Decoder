#include <transformer.h>
#include <iostream>
#include <torch/torch.h>

int main(){

    torch::Tensor tokens = torch::randint(30, {2,4});
    std::cout << "INPUT TOKENS\n" << tokens << "\n\n";
    
    std::shared_ptr<TransformerConfig> config = std::make_shared<TransformerConfig>();
    
    // embedding specs
    config->embedding_dim = 4;
    config->no_of_embeddings = 40;

    // positional encoding
    config->use_pre_calculated_pe = false;
    config->positional_encoder_precalculated_data_path = "/pe/pedata.pt";
    config->maximum_expected_range = 30;

    // decoder
    config->number_of_decoders = 1;

    // masked multihead attention specs
    config->wq_input_dim = 4;
    config->wq_output_dim = 4;
    config->wk_input_dim = 4;
    config->wk_output_dim = 4;
    config->wv_input_dim = 4;
    config->wv_output_dim = 4;

    // feed forward neural network specs
    config->ffnn_hidden_layer_dim = 8;
    try{
        Transformer transformer = Transformer(config);
        torch::Tensor data = transformer->forward(tokens);
        std::cout<<data<<"\n\n";
    }
    catch(const std::exception& e){
        std::cerr<<"exception\n\n"<<e.what()<<"\n";
    }

    

}