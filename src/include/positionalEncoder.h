#include <torch/torch.h>
#include <TransformerConfig.h>
#include <filesystem>
#include <stdexcept>

// expected input 
/*
{
    {
        {1, 2, 4, 5},
        {1, 2, 4, 5},
        {1, 2, 4, 5},
        {1, 2, 4, 5},
    },

    {
        {1, 2, 4, 5},
        {1, 2, 4, 5},
        {1, 2, 4, 5},
        {1, 2, 4, 5},
    },

}

    dim -> 3
    size -> 2 x 4 x 4
*/

// objective: The implementation should be efficient 

class PositionalEncoder {

private:

    std::string parameter_path;
    torch::Tensor pre_calculated_pe;


public:

    PositionalEncoder(const std::shared_ptr<TransformerConfig>& config, int expected_range_Y);

    static torch::Tensor EncodePosition(const int& y_range, const int& embedding_dim);

    // example usage
    /*
    auto tensor = torch::randint(
        1,20,
        {2,4,4}
    ).to(torch::kFloat);
    
    torch::Tensor x = PositionalEncoder::EncodePosition(tensor.size(-1), tensor.size(-2));
    torch::Tensor y =  x + tensor;
    */

    torch::Tensor EncodePositionP(const int& y_range, const int& embedding_dim);
    // this will load a prebuilt position and use this for better speed
    


};



