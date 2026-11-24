#pragma once

#include <vector>
#include <optional>
#include <map>
#include <ostream> 
#include <fstream>
#include <regex>
#include <string>
#include <iostream>
#include "json.hpp"

size_t count_elements(std::vector<size_t> shape);

struct Slice {
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    int64_t step = 1;
};



enum class DType {
    F32,
    F16,
    BF16,
    I8,
    I32
};

using Buffer = std::variant<
std::vector<float>
>;


size_t count_elements(std::vector<size_t> shape);
inline DType get_dtype(std::string dtype) {
    if (dtype == "BF16") return DType::BF16;
    if (dtype == "F16") return DType::F16;
    if (dtype == "I8") return DType::I8;
    if (dtype == "I32") return DType::I32;
    else return DType::F32;
}

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16:
        case DType::BF16: return 2;
        case DType::I8:  return 1;
        case DType::I32: return 4;
        default: return 0;
    }
}



