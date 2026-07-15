#include <decoder.h>
#include <attention.h>
#include <embedding.h>

DecoderImpl::DecoderImpl(const std::shared_ptr<TransformerConfig>& config){
    
    embedding_model = register_module(
        "embedding",
        Embedding(config)
    );
    attention_block = register_module(
        "attention",
        Attention(config)
    );
    normalization_block = register_module(
        "normalization1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config->embedding_dim})
        )
    );
    ffnn = register_module(
        "ffnn",
        FeedForwardNeuralNetwork(config)
    );

}

// 1 x decoder
// embedding block, positional encoding -> attention block -> residual add -> normalization -> feedforwardneuralnetwork -> residual add 
torch::Tensor DecoderImpl::forward(const torch::Tensor& input_tensor){

    // embedding block
    torch::Tensor embeddings = embedding_model->forward(input_tensor);

    // attention block
    torch::Tensor attention_output = attention_block->forward(embeddings);

    //residual add
    torch::Tensor residual_add = torch::add(input_tensor, attention_output);

    //layer normalization
    torch::Tensor normalized_val = normalization_block(residual_add);

    //feed forward neural network
    torch::Tensor ffnn_output = ffnn(normalized_val);

    //add residual
    return torch::add(input_tensor, ffnn_output);

}