
#include <stdexcept>
#include <span>
#include <string>
#include "safetensor.hpp"

using std::cout;
using std::endl;
using json = nlohmann::json;

Tensor::Tensor(std::vector<size_t> shape) : shape(shape) {
    int total_elements = 1;
    for (auto el : shape) { 
        total_elements *= el;
    }
    data = std::vector<float>(total_elements);
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
Tensor Tensor::operator*(int other) {
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

SafeTensors::SafeTensors(const std::string& path) {
    file = std::ifstream(path);
    char* buffer = new char [HEADER_SIZE];
    if (file) {
        std::cout << "Reading header size..." << std::endl;
        file.read(buffer, HEADER_SIZE);
    }
    // translating header information to unsigned 64
    unsigned long long header_un_long; 
    memcpy(&header_un_long, buffer, sizeof header_un_long);
    int header_size = (int) header_un_long;
    std::cout << "Json header size: " << header_size << std::endl;
    json_header_size = header_size;
    char* json_header = new char [header_size];

    if (file) {
        std::cout << "Reading json header..." << std::endl;
        file.read(json_header, header_size);
    }
    header = json::parse(json_header);
    
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor([";

    for (size_t i = 0; i < tensor.data.size(); ++i) {
        os << tensor.data[i];

        if (i + 1 < tensor.data.size()) {
            os << ", ";
        }
    }

    os << "])";
    return os;
}

Tensor SafeTensors::get_tensor(const std::string& name) {
    std::vector<uint64_t> offsets = header.at(name).at("data_offsets").get<std::vector<uint64_t>>();
    int start = offsets[0];
    int end = offsets[1];
    std::vector<size_t> shape = header.at(name).at("shape").get<std::vector<size_t>>();
    char* buffer = new char [end - start];
    file.seekg(0, file.beg);
    file.seekg(HEADER_SIZE + json_header_size + start, file.beg);
    Tensor tensor(shape);
    size_t num_elements = 1;

    for (size_t dim : shape) {
        num_elements *= dim;
    }

    std::vector<float> data(num_elements);

    file.read(
    reinterpret_cast<char*>(data.data()),
    data.size() * sizeof(float)
    );
    tensor.data = data;
    return tensor;
}


Tensor Tensor::operator()(std::vector<size_t> pos) {
    if (pos.size() > this->shape.size()) {
        std::cout << "Shape index out of range" << std::endl;
        throw std::invalid_argument("Invalid shape index, pos is larger then tensor shape");
    }

    for (int i = 0; i < pos.size(); ++i) {
        if (pos[i] >= shape[i])  {
            std::string msg = "Invalid shape index, index at dim " + i;
            msg += "out of range";
            throw std::invalid_argument(msg);
        }
    }
    size_t num_elements = data.size();
    size_t index = 0;
    for (int i = 0; i < pos.size(); ++i) {
        index += num_elements / shape[i] * pos[i];
        num_elements = num_elements /shape[i];
    }
    if (pos.size() == shape.size()) {
        Tensor result({data[index]}, {1});
        return result;

    } else {    
        std::span<float> new_data = std::span<float>(data).subspan(index, num_elements);
        std::span<size_t> new_shape = std::span<size_t>(shape).subspan(pos.size(), shape.size() - pos.size());
        std::vector<float> new_vec(new_data.size()); 
        std::vector<size_t> new_shape_vec(new_shape.size());
        new_vec.assign(new_data.begin(), new_data.end());
        new_shape_vec.assign(new_shape.begin(), new_shape.end());
        return Tensor(new_vec, new_shape_vec);
    }
}

Tensor matmul(Tensor& t1, Tensor& t2) {
    if (t1.shape[1] != t2.shape[0]) {
        std::cout << "Tensor shape does not match" << std::endl;
        return Tensor({0,0});
    }
    
    auto t1_shape = t1.shape;
    auto t2_shape = t2.shape;

    size_t N = t1_shape[0];
    size_t M = t2_shape[1];
    size_t K = t1_shape[1];

    Tensor res({N, M});
    float* a = t1.data.data();
    float* b = t2.data.data();
    float* c = res.data.data();
    const size_t ROW_BLOCK = 4;
    const size_t COL_BLOCK = 128;

    for (size_t row_block_index = 0; row_block_index < N; row_block_index += ROW_BLOCK) {
        for (size_t col_block_index = 0; col_block_index < K; col_block_index += COL_BLOCK) {
            const size_t row_limit = std::min(N, row_block_index + ROW_BLOCK);
            const size_t width = std::min(COL_BLOCK, K - col_block_index);

            for (size_t k =0; k < K; ++k) {
                // Loading row of B for update multiple rows in C
                float* b_row = b + k* M + col_block_index;

                for (size_t i = row_block_index; i < row_limit; ++i) {
                    float* c_row = c + i*M + col_block_index;
                    const float av = a[i * K + k];

                    for (size_t j = 0; j < width; ++j) {
                        c_row[j] += av * b_row[j];
                    }
                }
            }

            
        }
    }
    
    return res;
    
}

float& Tensor::at(size_t i, size_t j) {
    // if (shape.size() != 2) {
    //     throw std::runtime_error("Tensor must be size 2");
    // }
    // if (i >= shape[0] || j >= shape[1] || i*j < 0 || i * shape[1] + j >= data.size()) {
    //     std::ostringstream msg;
    //     msg << "Index " << i << ", " << j << " is out of range of Tensor shape: (" << shape[0] << ", " << shape[1] << ")" ;
    //     throw std::runtime_error(msg.str());
    // }
    if (i* shape[1] + j > data.size()) {
        throw std::runtime_error("Index out of bounds");
    }
    return data[i * shape[1] + j];
}



float& Tensor::item() {
    if (shape.size() != 1) {
        throw std::runtime_error("Tensor must be size 1");
    }
    if (shape[0] != 1) {
        throw std::runtime_error("Tensor must be size 1");
    }
    return data[0];
}

void Tensor::setItem(float item) {
    if (shape.size() != 1) {
        throw std::runtime_error("Tensor must be size 1");
    }
    if (shape[0] != 1) {
        throw std::runtime_error("Tensor must be size 1");
    }
    data[0] = item;
}