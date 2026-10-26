#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>
#include "safetensor.hpp"


class InferenceModel {

    public:
        InferenceModel(const std::string& name);
        int load_safetensor(const std::string& path);
        

    private:
        SafeTensor safetensor;
        
    };