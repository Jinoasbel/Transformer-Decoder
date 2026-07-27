#include <transformer.h>
#include <iostream>
#include <torch/torch.h>
#include <vector>

int main(){

    // torch::Tensor tokens = torch::randint(30, {2,4});
    // std::cout << "INPUT TOKENS\n" << tokens << "\n\n";
    
    std::shared_ptr<TransformerConfig> config = std::make_shared<TransformerConfig>();
    
    config->is_training_enabled = false;
    // embedding specs
    config->embedding_dim = 4;
    config->no_of_embeddings = 100;

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

    config->use_kv_cache = true;
    // feed forward neural network specs
    config->prediction_head_input = 4;
    config->prediction_head_output = 100;
    config->batch_size = 1;
    config->ffnn_hidden_layer_dim = 8;

    
    std::vector<torch::Tensor> t ;
    try{
        Transformer transformer = Transformer(config);
        torch::Tensor tokens = torch::randint(30, {1, 3}); // torch::randint(30, {1,1});
        // 1 x 1
        std::cout<<"INPUT TOKEN DIM "<<tokens.sizes()<<std::endl;
        std::cout<<tokens<<std::endl;

        for(int i = 0; i<5; i++){
            std::cout<<"__________ ITERATION __________ "<<i<<"\n\n";
            torch::Tensor data = transformer->forward(tokens);
            t.push_back(data);
            // tokens = data.unsqueeze(0);
            tokens = data;
            std::cout<<data<<"\n\n";
        }
        torch::Tensor data = transformer->forward(tokens);
        std::cout<<data<<"\n\n";
    }
    catch(const std::exception& e){
        std::cerr<<"exception\n\n"<<e.what()<<"\n";
    }

    for (torch::Tensor x : t){
        std::cout<<x<<std::endl;
    }

    

}