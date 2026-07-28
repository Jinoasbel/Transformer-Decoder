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
torch::Tensor PositionalEncoder::EncodePosition(const int& no_of_tokens, const int& embedding_dim, int token_no = 0){

    DEBUG("STATIC PE");

    token_no--;
    if(token_no > 0){
        
        auto position = torch::tensor({token_no}).unsqueeze(1);
        auto div = -std::log(10000)/embedding_dim;

        DEBUG("PE INFERENCE MODE");
        DEBUG(position);

        auto divisors = torch::exp(
            torch::arange(0, embedding_dim, 2) * div
        ); // e.[0, 2, 4,.., 2n] * div

        auto x = position * divisors;
    
        auto pe_table = torch::ones({embedding_dim});

        pe_table.index_put_(
            {torch::indexing::Slice(0, embedding_dim, 2)}, torch::sin(x)
        );
        pe_table.index_put_(
            {torch::indexing::Slice(1, embedding_dim, 2)}, torch::cos(x)
        );

        DEBUG("PE TABLE");
        DEBUG(pe_table);

        std::cerr<<"PE TABLE \n"<<pe_table<<std::endl;

        return pe_table;
    }
    auto y_position = torch::arange(no_of_tokens).unsqueeze(1); // makes [0,1,2,3..no_of_tokens-1] size = [no_of_tokens] or 1 dim => unsquueze(1)  [no_of_tokens, 1]
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
    auto x = y_position * divisors;

    // each will be common for cos(element x positon) or sin(element x positon) 
    
    auto pe_table = torch::ones({no_of_tokens, embedding_dim});

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

torch::Tensor PositionalEncoder :: EncodePositionP(const int& no_of_tokens){

    // this will load a prebuilt position and use this for better speed
    if (pre_calculated_pe.size(-2) < no_of_tokens){
        pre_calculated_pe = PositionalEncoder::EncodePosition(no_of_tokens, embedding_dim);
        torch::save(pre_calculated_pe, parameter_path);
    }
    DEBUG("PE_VALUES");    
    return pre_calculated_pe.index({torch::indexing::Slice(0, no_of_tokens), torch::indexing::Slice()});


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