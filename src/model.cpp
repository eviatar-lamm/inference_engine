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
            std::pair<std::vector<size_t>, std::vector<float>> tensor_vec = loader.get_tensor<float>(item.key());
            tensor_map.try_emplace(
                item.key() ,
                Tensor<float>(
        std::move(tensor_vec.first),
        std::move(tensor_vec.second)
    )
            );
        }
    }
    
    return 1;

}
void InferenceModel::list_items() {
    for (const auto& [key, val] : tensor_map) {
        std::cout << key << std::endl;
    }
}

GenTensor InferenceModel::get_layer(const std::string name) {
    if (tensor_map.contains(name)) {
    return tensor_map.at("layer.weight");
    } else {
        return Tensor<float>({});
    }
}




