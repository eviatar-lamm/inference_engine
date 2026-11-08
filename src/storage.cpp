
#include "storage.hpp"





Storage::Storage(std::unique_ptr<std::byte[]> data, DType dtype) : data_(std::move(data)) , dtype(dtype) {

}

template <typename T>
T& Storage::at(size_t index) {
    return reinterpret_cast<T*>(data_.get())[index];
}

template <typename T>
T* Storage::get_raw() {
    return reinterpret_cast<T*>(data_.get());
}