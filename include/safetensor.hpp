#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <json.hpp>









#include "tensor.hpp"



class SafeTensor {
public:
    SafeTensor(const std::string& path);

    template<typename T>
    std::pair<std::vector<size_t>, std::vector<T>> get_tensor(const std::string& name);

    nlohmann::json header;
private:
    DType check_dtype();
    std::ifstream file;
    int json_header_size;
    const int HEADER_SIZE = 8;
};

