#include <transformer.h>

TransformerImpl :: TransformerImpl(const std::shared_ptr<TransformerConfig>& config){

    DEBUG(config->number_of_decoders);
    emb_model = register_module(
        "embedding_model", Embedding(config)
    );


    // based on the transformer config create n decoder blocks and push each of them to collection_of_decoders
    for(int decoder = 0; decoder < config->number_of_decoders; decoder++){

        collection_of_decoders.push_back(
            register_module(
                "decoder_"+ std::to_string(decoder), Decoder(config)
            )
        );
        DEBUG("DECODER INIT");
    }

    DEBUG("INIT PREDICTION HEADS");
    prediction_heads = register_module(
        "prediction heads",
        Heads(config)
    );
    
}

// 1 x 1
torch::Tensor TransformerImpl :: forward(torch::Tensor& input){

    DEBUG("INPUT TOKEN");
    DEBUG(input);

    DEBUG("EMBEDDINGS");
    torch::Tensor embeddings = emb_model->forward(input);
    DEBUG(embeddings);
    DEBUG("DIMENSION OF EMBEDDING OUTPUT");
    DEBUG(embeddings.sizes());
    
    DEBUG("FORWARD"); 
    for(Decoder& decoder_i : collection_of_decoders){
        embeddings = decoder_i->forward(embeddings);
    }

    DEBUG("OUTPUT");
    DEBUG(embeddings);

    embeddings = prediction_heads->forward(embeddings);
    DEBUG("INDEX");
    DEBUG(embeddings);
    return embeddings;

}

