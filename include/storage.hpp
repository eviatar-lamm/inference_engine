
#pragma once
#include "utils.hpp"

class Storage {
    
    
    public:
        DType dtype = DType::F32;
        Storage(std::unique_ptr<std::byte[]> data, size_t size, DType dtype);
        Storage(size_t size, DType dtype);
        
        template <typename T>
        T& at(size_t index) {
            return reinterpret_cast<T*>(data_.get())[index];
        }

        template <typename T>
        T* get_raw() {
            return reinterpret_cast<T*>(data_.get());
        }

        size_t size() {
            return size_ / dtype_size(dtype);
        }

    private:
        std::unique_ptr<std::byte[]> data_;
        size_t size_;

};