#include <feedForwardNN.h>

FeedForwardNeuralNetworkImpl :: FeedForwardNeuralNetworkImpl(const std::shared_ptr<TransformerConfig>& config){
    layer1 = register_module(
        "FFNNlayer1",
        torch::nn::Linear(config->embedding_dim, config->ffnn_hidden_layer_dim)
    );

    layer2 = register_module(
        "FFNNlayer2",
        torch::nn::Linear(config->ffnn_hidden_layer_dim, config->embedding_dim)
    );
}

torch::Tensor FeedForwardNeuralNetworkImpl :: forward(const torch::Tensor& normalized_input){

    // embeddingdim -> expand -> shrink -> embeddingdim
    
    torch::Tensor layer1_output = layer1->forward(normalized_input);
    torch::Tensor activation_output = torch::gelu(layer1_output);
    return layer2->forward(layer1_output);
}