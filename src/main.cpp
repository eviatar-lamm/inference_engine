
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "safetensor.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

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
    const std::string MODEL_PATH = "./models/gpt2/model.safetensors";
    const std::string TOKENSIZER_PATH = "./models/gpt2/";

    // const std::string NAME = "hello";
    // InferenceModel model(NAME);
    // model.from_safetensor(MODEL_PATH);

    // model.list_items();
    
    Tokenizer t(TOKENSIZER_PATH);
    // t.bytes_to_unicode();
    // std::cout << ((size_t) *std::string("!").c_str() + 93 == (size_t) *std::string("~").c_str()) << "\n";
    std::string s = "Hello, what is your name?";
    std::vector<int> res = t.encode(s);
    for (auto t : res) std::cout << t << ", ";
    std::cout  << "\n";

    return 0;
}

