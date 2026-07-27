# include <heads.h>

HeadsImpl :: HeadsImpl(const std::shared_ptr<TransformerConfig>& config){

    DEBUG("PREDICTION HEADS INIT");
    prediction_heads = register_module(
        "prediction heads",
        torch::nn::Linear(config->prediction_head_input, config->prediction_head_output)
    );

    is_training_enabled = config->is_training_enabled;

}

torch::Tensor HeadsImpl :: forward(torch::Tensor& input){


    // This slices the last vector from the sequence if multiple tokens are fed at once
    if (input.size(-2) > 1 && !is_training_enabled){
        input = input.index(
            {
                torch::indexing::Slice(), 
                torch::indexing::Slice(-1, torch::indexing::None), 
                torch::indexing::Slice()
            });
    }
    // 1 x 1 x 4 -> 1 x 1 x 100
    torch::Tensor predictions = prediction_heads->forward(input);
    DEBUG("PREDICTIONS FORWARD");
    DEBUG(predictions);
    predictions = torch::softmax(predictions, -1);
    DEBUG("SOFT MAX");
    DEBUG(predictions);

    // 2 x 1 x 100
    // torch::Tensor index = predictions.select(1, predictions.size(1)-1);


    torch::Tensor index = predictions.argmax(-1); // 2 X 1 

    // {
    //     {x}
    //     {y}
    // }

    DEBUG("PREDICTED TOKEN");
    DEBUG(index);
    return index;

}