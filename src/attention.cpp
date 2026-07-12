#include <torch/torch.h>

class attention : public torch::nn::Module{

private:
    torch::nn::Linear wq {nullptr};
    torch::nn::Linear wk {nullptr};
    torch::nn::Linear wv {nullptr};

    torch::Tensor query, key, value, scores, mask;

public:
    attention(){

        wq = register_module(
            "wQ", 
            torch::nn::Linear(4, 4)
        );
        wk = register_module(
            "wK",
            torch::nn::Linear(4,4)
        );
        wv = register_module(
            "wV",
            torch::nn::Linear(4,4)
        );

    }

    void forward(const torch::Tensor& embeddings){

        query = wq->forward(embeddings);
        key = wk->forward(embeddings);
        value = wv->forward(embeddings);

        scores = torch::matmul(
            query, key.transpose(0,1)
        );


        mask = torch::triu(
            torch::full({10, 11}, -INFINITY),
            1
        );

        scores += mask;
        auto attention = torch::softmax(scores, -1);
        auto final_value = torch::matmul(attention, value);

    }


};