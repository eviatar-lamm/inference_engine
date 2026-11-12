
#include "storage.hpp"


Storage::Storage(std::unique_ptr<std::byte[]> data, size_t size, DType dtype) : 
    data_(std::move(data)) , size_(size), dtype(dtype) {

}

Storage::Storage(size_t size, DType dtype) : 
    size_(size), dtype(dtype) , data_(std::make_unique<std::byte[]>(size * dtype_size(dtype))) {
}

