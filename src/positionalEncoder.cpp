#include <positionalEncoder.h>

PositionalEncoder :: PositionalEncoder(const std::shared_ptr<TransformerConfig>& config) : pe_data_path(config->positional_encoder_precalculated_data_path), embedding_dim(config->embedding_dim){

    DEBUG("POSITIONAL ENCODER INIT ");
    if (std::filesystem::exists(pe_data_path)){     

        torch::load(pre_calculated_pe, pe_data_path);
        if(pre_calculated_pe.size(-1) != config->embedding_dim) throw std::runtime_error("embedding dimension mis-match during init from PositionalEncoding");

    }
    else{
        pre_calculated_pe = PositionalEncoder::EncodePosition(MAX_RANGE, config->embedding_dim);
        DEBUG("CREATING NEW PE BECAUSE PE NOT EXIST");
        torch::save(pre_calculated_pe, pe_data_path);
    }
}




torch::Tensor PositionalEncoder::EncodePosition(const int& no_of_tokens, const int& embedding_dim, int token_no){

    torch::Tensor position;
    torch::Tensor pos_div;
    torch::Tensor pe_table;

    DEBUG("STATIC PE");
    auto div = -std::log(10000)/embedding_dim;
    auto divisors = torch::exp(
        torch::arange(0, embedding_dim, 2) * div
    );

    token_no--;

    position = (token_no > 0)? torch::tensor({token_no}).unsqueeze(1) : torch::arange(no_of_tokens).unsqueeze(1);
    pe_table = (token_no > 0)? torch::ones({embedding_dim}).unsqueeze(0) : torch::ones({no_of_tokens, embedding_dim});

    pos_div = position * divisors;

    pe_table.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(0, embedding_dim, 2)}, torch::sin(pos_div)
    );
    pe_table.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(1, embedding_dim, 2)}, torch::cos(pos_div)
    );
    DEBUG("PE TABLE");
    DEBUG(pe_table);
    return pe_table;

}

torch::Tensor PositionalEncoder :: EncodePositionP(const int& no_of_tokens, int token_no){

    token_no--;
    if (pre_calculated_pe.size(-2) < no_of_tokens || pre_calculated_pe.size(-2) < token_no){
        DEBUG("CREATING NEW PE");
        pre_calculated_pe = PositionalEncoder::EncodePosition(
            (no_of_tokens != 0)? no_of_tokens : token_no + 1, embedding_dim
        );
        DEBUG("PE VALUES");
        DEBUG(pre_calculated_pe);
        torch::save(pre_calculated_pe, pe_data_path);
    }
    DEBUG("PE_VALUES");
    return (token_no > 0)? 
    pre_calculated_pe.index(
        {torch::indexing::Slice(token_no, token_no + 1), torch::indexing::Slice()}
    ) : pre_calculated_pe.index(
        {torch::indexing::Slice(0, no_of_tokens), torch::indexing::Slice()}
    );


}

