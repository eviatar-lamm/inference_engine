
#include <stdexcept>
#include <span>
#include <string>
#include "safetensor.hpp"
#include "tensor.hpp"


using std::cout;
using std::endl;
using json = nlohmann::json;



SafeTensor::SafeTensor(const std::string& path) {
    file = std::ifstream(path);
    char* buffer = new char [HEADER_SIZE];
    if (file) {
        std::cout << "Reading header size..." << std::endl;
        file.read(buffer, HEADER_SIZE);
    } else {
        std::cout << "Error reading file, check the PATH specifed: " << path << std::endl;
        return;
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

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
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



template <typename T>
std::pair<std::vector<size_t>,std::vector<T>> SafeTensor::get_tensor(const std::string& name) {
    std::vector<uint64_t> offsets = header.at(name).at("data_offsets").get<std::vector<uint64_t>>();
    int start = offsets[0];
    int end = offsets[1];

    std::vector<size_t> shape = header.at(name).at("shape").get<std::vector<size_t>>();
    DType dtype = get_dtype(header.at(name).at("dtype").get<std::string>());

    char* buffer = new char [end - start];
    file.seekg(0, file.beg);
    file.seekg(HEADER_SIZE + json_header_size + start, file.beg);
    size_t num_elements = count_elements(shape);

    std::vector<T> data(num_elements);

    file.read(
    reinterpret_cast<char*>(data.data()),
    data.size() * sizeof(T)
    );
    
    return std::pair<std::vector<size_t>,std::vector<T>>(shape, data);
}

template std::pair<
    std::vector<size_t>,
    std::vector<float>
>
SafeTensor::get_tensor<float>(
    const std::string&
);

auto matmul(auto& t1, auto& t2) {
    if (t1.shape[1] != t2.shape[0]) {
        std::cout << "Tensor shape does not match" << std::endl;
        return Tensor<float>({0,0});
    }
    
    auto t1_shape = t1.shape;
    auto t2_shape = t2.shape;

    size_t N = t1_shape[0];
    size_t M = t2_shape[1];
    size_t K = t1_shape[1];

    Tensor<float> res({N, M});
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



