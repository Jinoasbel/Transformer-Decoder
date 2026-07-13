# pragma once
#include <cstdint>

struct TransformerConfig{

    // embedding specs
    int64_t embedding_dim;
    int64_t no_of_embeddings;

    // masked multihead attention specs
    int64_t wq_input_dim;
    int64_t wq_output_dim;
    int64_t wk_input_dim;
    int64_t wk_output_dim;
    int64_t wv_input_dim;
    int64_t wv_output_dim;
    
};