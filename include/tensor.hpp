#pragma once
#include <vector>
#include <iostream>
#include <variant>


size_t count_elements(std::vector<size_t> shape);

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

auto create_tensor(std::vector<size_t>, std::string);

template <typename T>
class Tensor {
    public:
        Tensor(std::vector<size_t> shape, std::vector<T>);
        Tensor(std::vector<size_t> shape);
        std::vector<T> data;
        std::vector<size_t> shape;
        friend std::ostream& operator<<(std::ostream& os, const Tensor<float>& tensor);
        Tensor<T> operator+(Tensor<T>& other);
        Tensor<T> operator+(int other);
        Tensor<T> operator*(float);
        Tensor<T> operator*(Tensor<T>& other);
        Tensor<T> operator()(std::vector<size_t>);
        T& item();
        void setItem(T item);
};
 
