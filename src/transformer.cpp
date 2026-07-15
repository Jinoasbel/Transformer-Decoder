#include <transformer.h>

TransformerImpl :: TransformerImpl(const std::shared_ptr<TransformerConfig>& config){

    // based on the transformer config create n decoder blocks and push each of them to collection_of_decoders
    for(int decoder = 0; decoder < config->number_of_decoders; decoder++){
        collection_of_decoders.push_back(Decoder(config));
    }
}

torch::Tensor TransformerImpl :: forward(const torch::Tensor& input){

    torch::Tensor decoder_out = input;
    
    for(Decoder& decoder_i : collection_of_decoders){
        decoder_out = decoder_i->forward(decoder_out);
    }
}

