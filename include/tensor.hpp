#pragma once

#include "utils.hpp"
#include "storage.hpp"



auto create_tensor(std::vector<size_t>, std::string);


class Tensor;

template <typename T>
class TensorIterator;


class Tensor {
    public:
        std::vector<size_t> strides;
        std::vector<size_t> shape;
        DType dtype = DType::F32;
        size_t offset = 0;
        
        Tensor(std::vector<size_t> shape, std::shared_ptr<Storage>);
        Tensor(std::vector<size_t> shape);
        
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
        T& getItem(std::vector<size_t>);

        template <typename T>
        T* get_raw() {
            return storage_.get()->get_raw<T>();
        }

        void index_begin() {
            for (size_t i = 0; i < index_.size(); ++i) index_[i] = 0;
        }
        
        

        void flatten();

        size_t next() {
            size_t last = shape.size() - 1;
            bool overflow = false;
            while (true) {
                index_[last]++;
                if (index_[last] == shape[last]) overflow = true;
                if (overflow) {
                    for (int i = last; i > 0; --i) {
                        if (index_[i] == shape[i]) {
                            index_[i] = 0;
                            index_[i - 1]++;
                        } else break;

                    }
                    overflow = false; 
                    if (index_[0] == shape[0]) return 0;
                }
                // std::cout << index_[0] << ", " << index_[1] << " ," << index_[2] << std::endl;
                return get_raw_index(index_, *this);
            }


        }

        std::vector<size_t> index_end() {
            std::vector<size_t> end;
            for (auto dim : shape) end.push_back(--dim);
            return end;
        }

        template <typename T>
        TensorIterator<T> begin() {
            return TensorIterator<T>(this, std::vector<size_t>(shape.size()));
        }

        template <typename T>
        TensorIterator<T> end() {
            std::vector<size_t> last_index(shape.size());
            for (int i = 0; i < shape.size(); ++i) last_index[i] = --shape[i];
            return TensorIterator<T>(this, last_index);
        }

        size_t iter() {
            return next();      
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        Tensor operator+(Tensor& other);
        Tensor operator+(int other);
        Tensor operator*(float);
        Tensor operator*(Tensor& other);
        Tensor operator()(std::vector<size_t>);
        size_t size();

        
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

        TensorIterator& operator++() {
            index_[0]++;
            size_t i = tensor_.shape.size();
            if (index_[0] == tensor_.shape[0]) {
                while (index_[i] == tensor_.shape[i]) {
                    if (i == 0) return 0;
                    index_[i] == 0;
                    index_[i - 1]++;
                }
            }
            return *this;
        }
        bool operator==(const TensorIterator& other) {
            return this.tensor_ == other.tensor_ && index_ == other.index_;
        }

        bool operator!=(const TensorIterator& other) {
            return !(*this == other);
        }

        T& operator*() const {
            return tensor_.get_raw<T>() + tensor_.offset + Tensor::get_raw_index(index_);
        }

    private:
        Tensor& tensor_;
        std::vector<size_t> index_;
};

