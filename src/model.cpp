#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <json.hpp>

#include "tensor.hpp"
#include "safetensor.hpp"
#include "model.hpp"




InferenceModel::InferenceModel(const std::string path) : tokenizer_(path) {
    from_safetensor(path);
}

DType check_dtype() {
    return DType::F32;
}

int InferenceModel::from_safetensor(const std::string path) {
    SafeTensor loader(path + "/model.safetensors");
    for (auto item : loader.header.items()) {
        if (item.key() == "__metadata__") continue;
        Tensor tensor_vec = loader.get_tensor(item.key());
        tensor_map.try_emplace(
            item.key() ,
            tensor_vec
        );
        
    }
    std::ifstream file(path + "/config.json");
    std::stringstream buffer;

    buffer << file.rdbuf();
    config_ = json::parse(buffer.str());
    n_dim_ = config_.at("n_embd").get<int>();
    return 0;

}
void InferenceModel::list_items() {
    for (const auto& [key, val] : tensor_map) {
        std::cout << key << std::endl;
    }
}




// Returns tensor of the vectorized tokens, of shape (batch, dim)
Tensor InferenceModel::vectorize(const std::vector<int>& tokens) {
    Tensor t({tokens.size(), (size_t) n_dim_});
    
}



int InferenceModel::predict_next(const std::vector<int>& tokens) {
    
    return 0;
}

std::string InferenceModel::predict(std::string query) {
    std::vector<int> tokenized_query = tokenizer_.encode(query);
    int next_token = -1;
    while (next_token != tokenizer_.EOS) {
        next_token = predict_next(tokenized_query);
        tokenized_query.push_back(next_token);
    }

    return tokenizer_.decode(tokenized_query);
}

Tensor InferenceModel::get_layer(const std::string name) {
    if (tensor_map.contains(name)) {
    return tensor_map.at("layer.weight");
    } else {
        return Tensor({});
    }
}




