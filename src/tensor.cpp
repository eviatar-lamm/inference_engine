
#include "tensor.hpp"


template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape) : shape(shape), data(std::vector<T>(count_elements(shape))) {}


size_t count_elements(std::vector<size_t> shape) {
    size_t total = 1;
    for (auto el : shape) total *= el;
    return total;
}

// Tensor + Tensor
Tensor<T> Tensor::operator+(Tensor& other) {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }

    Tensor result(shape);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}

// Tensor + scalar
Tensor Tensor::operator+(int other) {
    Tensor result(shape);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other;
    }

    return result;
}

// Tensor * scalar
Tensor Tensor::operator*(int other) {
    Tensor result(shape);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other;
    }

    return result;
}

// Tensor * Tensor (element-wise)
Tensor Tensor::operator*(Tensor& other) {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }

    Tensor result(shape);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }

    return result;
}