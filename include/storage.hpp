

#pragma once
#include "tensor.hpp"


class Storage {
    
    
    public:
        DType dtype = DType::F32;
        Storage(std::unique_ptr<std::byte[]> data, DType dtype);

        template <typename T>
        T& at(size_t index);  
        
        template <typename T>
        T* get_raw();  

    private:
        std::unique_ptr<std::byte[]> data_;

};