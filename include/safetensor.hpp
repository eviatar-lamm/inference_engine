#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>










class Tensor {
    public:
        Tensor(std::vector<float> data, std::vector<size_t> shape) : data(data), shape(shape) {};
        Tensor(std::vector<size_t> shape);
        std::vector<float> data;
        std::vector<size_t> shape;
        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
        Tensor operator+(Tensor& other);
        Tensor operator+(int other);
        Tensor operator*(int);
        Tensor operator*(Tensor& other);
        Tensor operator()(std::vector<size_t>);
        float& item();
        float& at(size_t,size_t);
        void setItem(float item);
};
 
Tensor matmul(Tensor& t1, Tensor& t2);



class SafeTensors {
public:
    explicit SafeTensors(const std::string& path);

    Tensor get_tensor(const std::string& name);

private:
    std::ifstream file;
    nlohmann::json header;
    int json_header_size;
    const int HEADER_SIZE = 8;
};

