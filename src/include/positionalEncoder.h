#pragma once

#include <optional>
#include <torch/torch.h>
#include <transformerConfig.h>
#include <filesystem>
#include <stdexcept>

#define MAX_RANGE 10


/**
 * @brief Encode Positional data for transformer models, returns positional data either from the stored tensors or In-Time computation of 
 * positional data providing flexibility such as improved performance or improved memory/storage efficiency
 * @pre embedding dimension must be even
 * @returns torch::Tensor no_tokens, embedding_dim 
 * @post The return torch::Tensor should be added with the embedding torch::Tensor 
 * 
 * @author Jino Asbel
 * 
 * @note static encoding was the only method tested for functionality and other implementation isn't guarented to work good
 * @todo non static positonal encoder methods should be implemented in later release
 */


class PositionalEncoder {

private:

    const std::string pe_data_path;
    torch::Tensor pre_calculated_pe;
    const int embedding_dim;


public:

    
    PositionalEncoder(const std::shared_ptr<TransformerConfig>& config);

    
    /**
     * @param no_of_tokens PositionalEncoding that should be generated along the y axis
     * @param embedding_dim positionalEncoding that should be generated along the x axis
     * @param token_no (optional, 0) if token_no > 0 generate PositionalEncoding for the specific token
     * 
     * @return torch::Tensor of Positional Encoded Data
     * @test no failures where found
     * @date 2026-08-19 
     */
    static torch::Tensor EncodePosition(const int& no_of_tokens, const int& embedding_dim, int token_no = 0);

    /**
     * @todo Test, Reconstruct an object oriented performance oriented version of the positional encoder
     */

    torch::Tensor EncodePositionP(const int& no_of_tokens = 0, int token_no = 0);


};



