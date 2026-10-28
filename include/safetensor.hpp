#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>









#include "tensor.hpp"



class SafeTensor {
public:
    SafeTensor(const std::string& path);

    auto get_tensor(const std::string& name);

private:
    std::ifstream file;
    nlohmann::json header;
    int json_header_size;
    const int HEADER_SIZE = 8;
};

