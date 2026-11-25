#include "../include/tokenizer.hpp"

#include <stdexcept>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: tokenizer_tests <model_directory> <reference_json>\n";
        return 1;
    }

    // The tokenizer currently prints debug messages; test diagnostics use stderr.
    std::cout.setstate(std::ios::failbit);

    try {
        for (const auto* name : {"vocab.json", "merges.txt"}) {
            if (!std::ifstream(std::string(argv[1]) + "/" + name)) {
                throw std::runtime_error(std::string("Missing tokenizer file: ") + name);
            }
        }
        std::ifstream file(argv[2]);
        const auto reference = nlohmann::json::parse(file);
        const Tokenizer tokenizer(argv[1]);
        std::size_t passed = 0;

        for (const auto& test : reference.at("cases")) {
            const auto name = test.at("name").get<std::string>();
            const auto text = test.at("text").get<std::string>();
            const auto expected = test.at("ids").get<std::vector<int>>();
            try {
                const auto actual = tokenizer.encode(text);
                if (actual == expected) {
                    ++passed;
                    std::cerr << "PASS " << name << '\n';
                } else {
                    std::cerr << "FAIL " << name << " input=" << test.at("text")
                              << "\n  expected: " << test.at("ids")
                              << "\n  actual:   " << nlohmann::json(actual) << '\n';
                }
            } catch (const std::exception& error) {
                std::cerr << "FAIL " << name << " input=" << test.at("text")
                          << "\n  expected: " << test.at("ids")
                          << "\n  exception: " << error.what() << '\n';
            }
        }

        const auto total = reference.at("cases").size();
        std::cerr << passed << '/' << total << " English cases match OpenAI GPT-2\n";
        return passed == total ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Tokenizer test setup failed: " << error.what() << '\n';
        return 1;
    }
}
