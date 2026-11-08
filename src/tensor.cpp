#include "tensor.hpp"
#include "storage.hpp"



// Tensor::Tensor(std::vector<size_t> shape) : shape(shape), data(std::vector(count_elements(shape))) {}


Tensor::Tensor(std::vector<size_t> shape, std::shared_ptr<Storage> storage) : shape(shape) , storage_(std::move(storage)) {
    for (int i = 1; i < shape.size() - 1; ++i) {
    size_t stride_i = shape[i];

        for (int j = i + 1; j < shape.size() - 1; ++j) {
            stride_i *= shape[j];
        }
        strides.push_back(stride_i);        
    }
}

size_t count_elements(std::vector<size_t> shape) {
    size_t total = 1;
    for (auto el : shape) total *= el;
    return total;
}


Tensor Tensor::operator()(std::vector<size_t> index) {
    
}


template <typename T>
T& Tensor::item() {
    switch(dtype) {
        case(DType::F32):
            storage_.get()->at()
    }
}

// Tensor + Tensor

Tensor Tensor::operator+(Tensor& other) {
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

Tensor Tensor::operator*(float other) {
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

