#include <iostream>
#include <unistd.h>

#include "tensor.hpp"
#include "storage.hpp"



Tensor::Tensor(std::vector<size_t> shape) : shape(shape) , offset(0){
    storage_ = std::make_shared<Storage>(count_elements(shape), DType::F32);
    strides = calulate_strides(shape);

}


Tensor::Tensor(std::vector<size_t> shape, std::shared_ptr<Storage> storage) : shape(shape) , storage_(std::move(storage)), offset(0) {
    strides = calulate_strides(shape);
    dtype = storage_.get()->dtype;
}


//calculate strides by given shape
std::vector<size_t> calulate_strides(std::vector<size_t> shape) {
    std::vector<size_t> strides;

    
    for (int i = 1; i < shape.size(); ++i) {
        size_t stride_i = 1;

        for (int j = i; j < shape.size(); ++j) {
            stride_i *= shape[j];
        }
        strides.push_back(stride_i);
    }
    strides.push_back(1);  
    return strides;
}

size_t count_elements(std::vector<size_t> shape) {
    size_t total = 1;
    for (auto el : shape) total *= el;
    return total;
}


size_t Tensor::get_raw_index(std::vector<size_t> index, Tensor& tensor) {
    size_t raw_index = 0;
    if (index.size() > tensor.shape.size()) return raw_index;
    for (size_t i = 0; i < index.size(); ++i) if (index[i] >= tensor.shape[i]) return 0; else raw_index += tensor.strides[i] * index[i];
    return raw_index + tensor.offset;
}



Tensor Tensor::operator()(std::vector<size_t> index) {
    // Calculating new Tensor shape
    std::vector<size_t> new_shape;
    if (index.size() == shape.size()) new_shape.push_back(1);
    else for (size_t i = index.size(); i < shape.size(); ++i) new_shape.push_back(shape[i]);

    Tensor new_tensor(new_shape, storage_);
    new_tensor.offset += Tensor::get_raw_index(index, *this) + offset;
    return new_tensor;
}





size_t Tensor::size() {
    return storage_.get()->size();
}
// Tensor + Tensor





Tensor Tensor::operator+(Tensor& other) {
    if (size() != other.size()) {   
        throw std::invalid_argument("Tensor sizes do not match");
    }

    Tensor result(shape);
    switch(dtype) {
        case(DType::F32): {
            auto i = this->begin<float>();
            auto res_i = result.begin<float>();
            auto other_i = other.begin<float>();

            while (i != this->end<float>()) {
                // std::cout << &operator()({1,1,4}).item<float>() << " " << &*i << std::endl;
                *res_i = *i + *other_i;
                ++res_i;
                ++i;
                ++other_i;
            }
            break;
            }
        default:
            return Tensor(shape);
    }
    return result;
    
    }

// Tensor + scalar
Tensor Tensor::operator+(int other) {
    Tensor result(shape);

    switch(dtype) {
        case(DType::F32):{
            auto i = this->begin<float>();
            auto res_i = result.begin<float>();

            while (i != this->end<float>()) {
                *res_i = *i + other;
                ++res_i;
                ++i;
            }
            break;}
        default:
            return Tensor(shape);
    }

    return result;
}

// Tensor * scalar

Tensor Tensor::operator*(float other) {
    Tensor result(shape);
    switch(dtype) {
        case(DType::F32):{
            auto i = this->begin<float>();
            auto res_i = result.begin<float>();

            while (i != this->end<float>()) {
                *res_i = *i * other;
                ++res_i;
                ++i;
            }
            break;}
        default:
            return Tensor(shape);
    }

    return result;
}





// Tensor * Tensor (element-wise)

Tensor Tensor::operator*(Tensor& other) {
    if (size() != other.size()) {
        throw std::invalid_argument("Tensor sizes do not match");
    }

    Tensor result(shape);
    switch(dtype) {
        case(DType::F32):{
            auto i = this->begin<float>();
            auto res_i = result.begin<float>();
            auto other_i = other.begin<float>();

            while (i != this->end<float>()) {
                *res_i = *i * *other_i;
                ++res_i;
                ++i;
                ++other_i;
            } break;
        }
        default:
            return Tensor(shape);
            
    }

    return result;
}


