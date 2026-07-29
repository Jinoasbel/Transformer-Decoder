#pragma once

#include <optional>
#include <torch/torch.h>
#include <transformerConfig.h>
#include <filesystem>
#include <stdexcept>

#define MAX_RANGE 10


class PositionalEncoder {

private:

    const std::string pe_data_path;
    torch::Tensor pre_calculated_pe;
    const int embedding_dim;


public:

    PositionalEncoder(const std::shared_ptr<TransformerConfig>& config);

    static torch::Tensor EncodePosition(const int& y_range, const int& embedding_dim, int token_no = 0);
        // if token_no = 0
        // return pe of size no_of_tokens X embedding_dim
        // else 
        // return pe of size 1 X embedding_dim

    torch::Tensor EncodePositionP(const int& no_of_tokens = 0, int token_no = 0);
        // returns precalculated positional encodings
        // args token_counts return token_counts pe
        // args 0, token_no returns token_no pe
    


};



