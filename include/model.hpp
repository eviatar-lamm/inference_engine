#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <json.hpp>
#include <unordered_map>
#include "safetensor.hpp"





using GenTensor = std::variant<
Tensor<float>>;


class InferenceModel {

    public:
        InferenceModel(const std::string name);
        int from_safetensor(const std::string path);
        std::unordered_map<std::string, GenTensor> tensor_map;
        std::string name;
        GenTensor get_layer(const std::string name);
        void list_items();


        
};