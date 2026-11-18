
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "safetensor.hpp"
#include "model.hpp"

using std::vector;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;


void print_vector(std::vector<int> v) {
    for (auto e : v) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}



void print_vector(std::vector<size_t> v) {
    for (auto e : v) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print_arr(T* arr, int size) {
    for (int i = 0; i < size; ++i) std::cout << *arr++ << " ";
    std::cout << std::endl;
}

void print_tensor(Tensor& t) {
    for (auto i = t.begin<float>(); i != t.end<float>(); ++i) std::cout << *i << " ";
    std::cout << std::endl;
}

int main () {
    // const std::string MODEL_PATH = "./models/gpt2/model.safetensors";
    // const std::string NAME = "hello";
    // InferenceModel model(NAME);
    // model.from_safetensor(MODEL_PATH);

    // model.list_items();

    Tensor a({});
    a.item<float>() = 10;
    std::vector<int> b(10);
    std::cout << std::vector<int>{1}.size();
    for (auto i = a.begin<float>(); i != a.end<float>(); ++i) std::cout << *i << " ";
    return 0;
}

