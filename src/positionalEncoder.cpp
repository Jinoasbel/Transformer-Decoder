#include <positionalEncoder.h>

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

// returns the positional encoding table 
// add the pe value with the embedding
torch::Tensor PositionalEncoder::EncodePosition(const int& y_range, const int& embedding_dim){

    DEBUG("STATIC PE");
    auto y_position = torch::arange(y_range).unsqueeze(1); // makes [0,1,2,3..y_range-1] size = [y_range] or 1 dim => unsquueze(1)  [y_range, 1]
    /*

    [1,2,3,4] would become

    [
        [1]
        [2]
        [3]
        [4]
    ]

    */
   // e−(2i/d)ln(10000)=e(2i)(d−ln(10000)​)

   auto div = -std::log(10000)/embedding_dim;

    auto divisors = torch::exp(
        torch::arange(0, embedding_dim, 2) * div
    );

    // makes divisor so we need to only just multiply the position with divisor
    // sin(x) or cos(x) where x = y_position/divisor
    auto x = y_position / divisors;

    // each will be common for cos(element x positon) or sin(element x positon) 
    
    auto pe_table = torch::ones({y_range, embedding_dim});

    pe_table.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(0, embedding_dim, 2)}, torch::sin(x)
    );
    // put new elements over the even dim

    pe_table.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(1, embedding_dim, 2)}, torch::cos(x)
    );
    // put new elements over the odd dim
    DEBUG("PE TABLE");
    DEBUG(pe_table);

    return pe_table;
}

torch::Tensor PositionalEncoder :: EncodePositionP(const int& y_range){

    // this will load a prebuilt position and use this for better speed
    if (pre_calculated_pe.size(-2) < y_range){
        pre_calculated_pe = PositionalEncoder::EncodePosition(y_range, embedding_dim);
        torch::save(pre_calculated_pe, parameter_path);
    }
    DEBUG("PE_VALUES");    
    return pre_calculated_pe.index({torch::indexing::Slice(0, y_range), torch::indexing::Slice()});


}

PositionalEncoder :: PositionalEncoder(const std::shared_ptr<TransformerConfig>& config) : parameter_path(config->positional_encoder_precalculated_data_path), embedding_dim(config->embedding_dim){

    DEBUG("PE INIT ");
    if (std::filesystem::exists(parameter_path)){     

        torch::load(pre_calculated_pe, parameter_path);
        if(pre_calculated_pe.size(-1) != config->embedding_dim) throw std::runtime_error("embedding dimension mis-match during init from PositionalEncoding");

    }
    else{
        pre_calculated_pe = PositionalEncoder::EncodePosition(config->maximum_expected_range, config->embedding_dim);
        DEBUG("CREATING NEW PE BECAUSE PE NOT EXIST");
        torch::save(pre_calculated_pe, parameter_path);
    }
}