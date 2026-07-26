#include <torch/torch.h>
#include <transformerConfig.h>

class HeadsImpl : public torch::nn::Module{

private:
    torch::nn::Linear prediction_heads {nullptr};
public:
    HeadsImpl(const std::shared_ptr<TransformerConfig>& config);
    torch::Tensor forward(const torch::Tensor& input);
};
TORCH_MODULE(Heads);