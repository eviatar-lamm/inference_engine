#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <json.hpp>

#include "tensor.hpp"
#include "safetensor.hpp"
#include "model.hpp"





InferenceModel::InferenceModel(const std::string name) : name(name) {

}


DType check_dtype() {
    return DType::F32;
}

int InferenceModel::from_safetensor(const std::string path) {
    SafeTensor loader(path);
    if (check_dtype() == DType::F32) {
        for (auto item : loader.header.items()) {
            if (item.key() == "__metadata__") continue;
            Tensor tensor_vec = loader.get_tensor(item.key());
            tensor_map.try_emplace(
                item.key() ,
                tensor_vec
            );
        }
    }
    
    return 0;

}
void InferenceModel::list_items() {
    for (const auto& [key, val] : tensor_map) {
        std::cout << key << std::endl;
    }
}



std::string predict_next(std::string& query) {
    return "";
}

Tensor InferenceModel::get_layer(const std::string name) {
    if (tensor_map.contains(name)) {
    return tensor_map.at("layer.weight");
    } else {
        return Tensor({});
    }
}




