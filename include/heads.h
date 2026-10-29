#include <torch/torch.h>
#include <transformerConfig.h>

class HeadsImpl : public torch::nn::Module{

private:
    torch::nn::Linear prediction_heads {nullptr};
    bool is_training_enabled;
public:
    HeadsImpl(const std::shared_ptr<TransformerConfig>& config);
    torch::Tensor forward(torch::Tensor& input);
};
TORCH_MODULE(Heads);