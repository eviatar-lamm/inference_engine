#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <json.hpp>
#include <unordered_map>
#include "safetensor.hpp"








class InferenceModel {

    public:
        InferenceModel(const std::string name);
        int from_safetensor(const std::string path);
        std::unordered_map<std::string, Tensor> tensor_map;
        std::string name;
        Tensor get_layer(const std::string name);
        void list_items();   
};