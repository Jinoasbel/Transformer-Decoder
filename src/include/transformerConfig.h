# pragma once
#include <cstdint>
#include <string>

#define DEBUG_ENABLED true

#if DEBUG_ENABLED
#define DEBUG(x) std::cout <<x<<"\n__________________________________________________________________\n\n"
#else
#define DEBUG(X) 
#endif

struct TransformerConfig{

    // Transformer configuration or settings
    bool is_training_enabled; // 0 if inference 1 if training 

        // embedding block
            // embedding specs
                int64_t embedding_dim;
                int64_t no_of_embeddings;

            // positional encoding
                bool use_pre_calculated_pe;
                std::string positional_encoder_precalculated_data_path;
                int64_t maximum_expected_range;

        // decoder block
            int number_of_decoders;

                // masked multihead attention specs
                    int64_t wq_input_dim;
                    int64_t wq_output_dim;
                    int64_t wk_input_dim;
                    int64_t wk_output_dim;
                    int64_t wv_input_dim;
                    int64_t wv_output_dim;

                // kv cache
                    bool use_kv_cache;


                // feed forward neural network specs
                int64_t ffnn_hidden_layer_dim;

    
    
};