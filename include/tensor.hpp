#pragma once

#include <iostream>
#include <memory>
#include <cmath>

#include "utils.hpp"
#include "storage.hpp"



class Tensor;

template <typename T>
class TensorIterator;


class Tensor {
    public:
        std::vector<size_t> strides;
        std::vector<size_t> shape;
        DType dtype = DType::F32;
        size_t offset = 0;
        bool is_empty = false;
        Tensor(std::vector<size_t> shape, std::shared_ptr<Storage>);
        Tensor(std::vector<size_t> shape);

        Tensor slice(size_t dim, Slice slice);
        
        template <typename T>
        T& item() {
            switch(dtype) {
                case(DType::F32):
                    return storage_.get()->at<float>(offset);
                default:
                    return storage_.get()->at<float>(0);
            }
        }

        template <typename T>
        T* get_raw() {
            return storage_.get()->get_raw<T>();
        }

        template <typename T>
        TensorIterator<T> begin() {
            return TensorIterator<T>(*this, std::vector<size_t>(shape.size()));
        }

        template <typename T>
        TensorIterator<T> end() {
            std::vector<size_t> last_index(shape.size());
            if (shape.size() > 0) last_index[0] = shape[0];
            else {
                TensorIterator<T> it(*this, last_index);
                it.is_scalar = true;
                return it;
            }
            return TensorIterator<T>(*this, last_index);
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        Tensor operator+(Tensor& other);
        Tensor operator+(int other);
        Tensor operator*(float);
        Tensor operator*(Tensor& other);
        Tensor operator()(std::vector<size_t>);
        size_t size();

        template <typename... Args>
            requires (std::same_as<Args, Slice> && ...)
        Tensor operator[](Args ... args) {

            std::array<Slice, sizeof...(Args)> values{
            static_cast<Slice>(args)...
            };

            if (values.size() > shape.size()) {
                std::cout << "Too many indexies, given " << values.size() << " slices, for Tensor with shape size of " << shape.size() << std::endl;
                return Tensor({});
            }
            size_t res_offset = 1;
            size_t total_offest = 0;
            for (int i = shape.size() - 1; i >= 1; --i) res_offset *= shape[i];

            std::vector<size_t> res_shape = shape;
            std::vector<size_t> res_strides = strides;
            res_shape = shape;
            size_t dim = 0;

            for (Slice slice : values) {

                float start = 0;
                float end = shape[dim] - 1;
                if (slice.start.has_value()) start = slice.start.value();
                if (slice.stop.has_value()) end = slice.stop.value() > end ? end : slice.stop.value();
                
                // Constructing the output tensor shape
                res_shape[dim] = std::ceil((end - start + 1)/ (float) slice.step);
                res_strides[dim] *= slice.step;
                
                total_offest += strides[dim] * start;
                
                dim++;
            }
            Tensor res(res_shape, storage_);
            res.strides = res_strides;
            std::cout << "total: " << total_offest << " paretnt: " << offset << std::endl;
            res.offset = total_offest + offset;
        return res;
        }

        static size_t get_raw_index(std::vector<size_t> index, Tensor&);
    private:
        std::shared_ptr<Storage> storage_;
        bool is_contigouse = false;
        std::vector<size_t> index_ = std::vector<size_t>(shape.size());

};



std::vector<size_t> calulate_strides(std::vector<size_t> shape);


template <typename T>
class TensorIterator {
    public:
    TensorIterator(Tensor& tensor, std::vector<size_t> position) : tensor_(tensor), index_(position) {}
    bool is_scalar = false;
    
        TensorIterator& operator++() {
            if (tensor_.shape.size() == 0) {
                is_scalar = true;
                return *this;
            }
            size_t last = index_.size() - 1 >= 0 ? index_.size() - 1 : 0;
            ++index_[last];
            size_t i = last;
            if (index_[last] == tensor_.shape[last]) {
                while (index_[i] == tensor_.shape[i]) {
                    if (i == 0) {
                        return *this;
                    }
                    index_[i] = 0;
                    index_[i - 1]++;
                    i--;
                }
            }
            return *this;
        }
        bool operator==(const TensorIterator& other) {
            return (&tensor_ ==  &other.tensor_) && (index_ == other.index_) && (is_scalar == other.is_scalar);
        }

        bool operator!=(const TensorIterator& other) {
            return !(*this == other);
        }

        T& operator*() const {
            return *(tensor_.get_raw<T>() + Tensor::get_raw_index(index_, tensor_));
        }

        std::vector<size_t> index_;
    private:
        Tensor& tensor_;
};

