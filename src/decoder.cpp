#include <decoder.h>


DecoderImpl::DecoderImpl(const std::shared_ptr<TransformerConfig>& config){
    
    DEBUG("INIT DECODER");
    attention_block = register_module(
        "attention",
        Attention(config)
    );
    DEBUG("ATTENTION INIT");
    normalization_block = register_module(
        "normalization1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config->embedding_dim})
        )
    );
    DEBUG("NORMALIZATION INIT");
    ffnn = register_module(
        "ffnn",
        FeedForwardNeuralNetwork(config)
    );
    DEBUG("FEED FORWARD NEURAL NETWORK INIT");

}

// 1 x decoder
// embedding block, positional encoding -> attention block -> residual add -> normalization -> feedforwardneuralnetwork -> residual add 
torch::Tensor DecoderImpl::forward(const torch::Tensor& input_tensor){

    DEBUG("DECODER FORWARD PASS");
    DEBUG(input_tensor);

    torch::Tensor attention_output = attention_block->forward(input_tensor);
    DEBUG("ATTENTION OUT");
    DEBUG(attention_output);

    //residual add
    torch::Tensor residual_add = torch::add(input_tensor, attention_output);
    DEBUG("RESIDUAL ADD");
    DEBUG(residual_add);

    //layer normalization
    torch::Tensor normalized_val = normalization_block(residual_add);
    DEBUG("NORMALIZATION");
    DEBUG(normalized_val);

    //feed forward neural network
    torch::Tensor ffnn_output = ffnn(normalized_val);
    DEBUG("FEED FORWARD NEURAL NETWORK");
    DEBUG(ffnn_output);
    
    //add residual
    return torch::add(input_tensor, ffnn_output);

}