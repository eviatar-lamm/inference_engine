
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


int main () {
    const std::string MODEL_PATH = "./models/gpt2/model.safetensors";
    const std::string NAME = "hello";
    InferenceModel model(NAME);
    model.from_safetensor(MODEL_PATH);

    model.list_items();

}

