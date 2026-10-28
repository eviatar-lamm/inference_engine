#include <stdfloat>
#include "tensor.hpp"



template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape) : shape(shape), data(std::vector<T>(count_elements(shape))) {}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, void* ptr) : shape(shape) , data(std::vector<T>(ptr, ptr+ count_elements(shape))) {}

size_t count_elements(std::vector<size_t> shape) {
    size_t total = 1;
    for (auto el : shape) total *= el;
    return total;
}

auto create_tensor(std::vector<size_t> shape, std::string& dtype) {
    if (dtype == "F32") return Tensor<float>(shape);
    return Tensor<float>(shape);
}

auto create_tensor_from_ptr(std::vector<size_t> shape, void* ptr, std::string& dtype) {
    if (dtype == "F32") return Tensor<float>(shape, ptr);
    return Tensor<float>(shape);
}

// Tensor + Tensor
template <typename T>
Tensor<T> Tensor<T>::operator+(Tensor& other) {
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
template <typename T>
Tensor<T> Tensor<T>::operator+(int other) {
    Tensor<T> result(shape);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other;
    }

    return result;
}

// Tensor<T> * scalar
template <typename T>
Tensor<T> Tensor<T>::operator*(float other) {
    Tensor<T> result(shape);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other;
    }

    return result;
}

// Tensor<T> * Tensor<T> (element-wise)
template <typename T>
Tensor<T> Tensor<T>::operator*(Tensor<T>& other) {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Tensor<T> sizes do not match");
    }

    Tensor<T> result(shape);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }

    return result;
}