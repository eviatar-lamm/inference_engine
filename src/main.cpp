
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "safetensor.hpp"


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
    SafeTensors ten(MODEL_PATH);
    Tensor test =  ten.get_tensor("h.9.mlp.c_proj.weight");
    
    std::cout << "Test tensor shape: ";
    print_vector(test.shape);
    std::cout << std::endl;


    Tensor a({3072,378});
    Tensor b({378,3072});



    auto start = std::chrono::steady_clock::now();

    auto result = matmul(a, b);

    auto end = std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Time: " << seconds << " s\n";

}

