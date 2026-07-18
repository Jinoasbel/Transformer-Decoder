# pragma once
#include <cstdint>
#include <string>

struct TransformerConfig{



        // embedding specs
        int64_t embedding_dim;
        int64_t no_of_embeddings;

        // positional encoding
        std::string positional_encoder_precalculated_data_path;

    //decoder
    int number_of_decoders;

        // masked multihead attention specs
        int64_t wq_input_dim;
        int64_t wq_output_dim;
        int64_t wk_input_dim;
        int64_t wk_output_dim;
        int64_t wv_input_dim;
        int64_t wv_output_dim;

        // feed forward neural network specs
        int64_t ffnn_hidden_layer_dim;

    
    
};