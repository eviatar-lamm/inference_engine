#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <json.hpp>
#include <unordered_map>

#include "tokenizer.hpp"
#include "safetensor.hpp"





using json = nlohmann::json;


class InferenceModel {

    public:
        InferenceModel(const std::string path);
        int from_safetensor(const std::string path);
        std::unordered_map<std::string, Tensor> tensor_map;
        std::string name;
        Tensor get_layer(const std::string name);
        void list_items();   
        std::string predict(std::string);
        int predict_next(const std::vector<int>&);
    
    private:
        Tokenizer tokenizer_;
        Tensor vectorize(const std::vector<int>&);
        json config_;
        int n_dim_;
};