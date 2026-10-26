#pragma once
#include <vector>
#include <iostream>

enum class DType {
    F32,
    F16,
    BF16,
    I8,
    I32
};

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
template <typename T>
class Tensor {
    public:
        Tensor(void* data, std::vector<size_t> shape);
        Tensor(std::vector<size_t> shape);
        std::vector<T> data;
        std::vector<size_t> shape;
        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        Tensor<T> operator+(Tensor<T>& other);
        Tensor<T> operator+(int other);
        Tensor<T> operator*(float);
        Tensor<T> operator*(Tensor<T>& other);
        Tensor<T> operator()(std::vector<size_t>);
        T& item();
        void setItem(T item);
};
 
