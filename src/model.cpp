#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>
#include "safetensor.hpp"
#include "model.hpp"



int InferenceModel::load_safetensor(const std::string& path) {

    safetensor = SafeTensor(path);

}

