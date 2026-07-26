# include <heads.h>

HeadsImpl :: HeadsImpl(const std::shared_ptr<TransformerConfig>& config){

    DEBUG("PREDICTION HEADS INIT");
    prediction_heads = register_module(
        "prediction heads",
        torch::nn::Linear(config->prediction_head_input, config->prediction_head_output)
    );

}

torch::Tensor HeadsImpl :: forward(const torch::Tensor& input){

    torch::Tensor predictions = prediction_heads->forward(input);
    DEBUG("PREDICTIONS FORWARD");
    DEBUG(predictions);
    predictions = torch::softmax(predictions, -1);
    DEBUG("SOFT MAX");
    DEBUG(predictions);

    torch::Tensor index = predictions.select(1, predictions.size(1)-1);
    index = index.argmax(-1);
    DEBUG("PREDICTED TOKEN");
    DEBUG(index);
    return index;

}