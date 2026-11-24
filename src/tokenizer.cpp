#include <limits>
#include <sstream>

#include "tokenizer.hpp"

using json = nlohmann::json;


void Tokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    std::stringstream buffer;

    buffer << file.rdbuf();
    std::cout << buffer.str().size() << "\n";
    nlohmann::json vocab = json::parse(buffer.str());
    
    for (auto pair : vocab.items()) vocab_map.try_emplace(pair.key(), pair.value());
}

void Tokenizer::load_merge(const std::string& path) {

    std::ifstream file(path);
    std::string line;
    int priority = 1;
    while (std::getline(file, line)) {
        int space = line.find(" ");
        
        // Parsing the merge line
        std::string first = line.substr(0, space);
        std::string second = line.substr(space + 1, line.size());
        
        merge_map.try_emplace({first, second}, priority);
        
        ++priority;
    }
}

Tokenizer::Tokenizer(const std::string& path) {
    const std::string MERGE_PATH = path + "/merges.txt";
    const std::string VOCAB_PATH = path + "/vocab.json";
    load_merge(MERGE_PATH);
    load_vocab(VOCAB_PATH);
}



// Pre process the use text input, into GPT2 pre tokenized symbols
std::vector<std::string> Tokenizer::pre_tokenize(const std::string& query) const {
    std::vector<std::string> token_array;
    try
        {
            const std::regex REG("'(?:s|t|re|ve|m|ll|d)| ?\\w+| ?\\d+| ?[^\\s\\w\\d]+|\\s+(?!\\S)|\\s+");
            auto words_begin = std::sregex_iterator(query.begin(), query.end(), REG);
            auto workds_end = std::sregex_iterator();
            for (std::sregex_iterator i = words_begin; i != workds_end; ++i) {
                token_array.push_back((*i).str());
            } 
            
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    return token_array;
}



std::string to_utf8(char32_t cp) {
    std::string result;

    if (cp <= 0x7F) {
        result += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }

    return result;
}


std::map<size_t,size_t> Tokenizer::bytes_to_unicode() const {
    /*
    Implementation of bytes_to_unicode function of GPT2 itself.
    */
    std::vector<size_t> bytes;
    std::map<size_t, size_t> bytes_to_unicode_map;
    for (size_t i = static_cast<size_t>(U'!'); i <= static_cast<size_t>(U'~'); ++i) bytes.push_back(i);
    for (size_t i = static_cast<size_t>(U'¡'); i <= static_cast<size_t>(U'¬'); ++i) bytes.push_back(i);
    for (size_t i = static_cast<size_t>(U'®'); i <= static_cast<size_t>(U'ÿ'); ++i) bytes.push_back(i);
    
    std::vector<size_t> gpt_bytes = bytes;
    
    size_t n = 0;
    for (size_t i = 0; i < (size_t) pow(2,8); ++i) {
        if (!std::count(bytes.begin(), bytes.end(), i)) {
            bytes.push_back(i);
            gpt_bytes.push_back((size_t) pow(2,8) + n);
            n++;
        } 
    }
    
    for (int i = 0; i < bytes.size(); ++i) bytes_to_unicode_map[bytes[i]] = gpt_bytes[i];
    return bytes_to_unicode_map;
}




// Returns the rank of certain pair of symbols, by the merge.txt file
int Tokenizer::get_rank(std::vector<std::string>& pair) const {
    auto it = merge_map.find(pair);

    if (it != merge_map.end())
        return it->second;

    return -1;
}

// Returns the indexes of the lowest pair in the symbol array
std::vector<int> Tokenizer::get_lowest_pair(std::vector<std::string>& symbols) const {
    std::vector<int> lowest_pair{0, 0};
    int lowest_rank = std::numeric_limits<int>::max();;
    for (int i = 0; i < symbols.size() - 1; ++i) {
        int j = i + 1;
        std::vector<std::string> current_pair = {symbols[i], symbols[j]};
        // std::cout << current_pair[0] << " " << current_pair[1] << "\n";
        int current = get_rank(current_pair);
        if (current == -1) continue;
        if (current < lowest_rank) {
            // Well, Obvious
            lowest_rank = current;
            lowest_pair[0] = i;
            lowest_pair[1] = j;
            
        }
    }
    return lowest_pair;
}



void Tokenizer::bpe_merge(std::vector<std::string>& symbols) const{
    
    while(true) {
        std::cout << "The array now: ";
        // for (auto d : symbols) std::cout << d << ", ";
        // std::cout << "\n";
        std::vector<int> lowest_pair = get_lowest_pair(symbols);
        // std::cout << "The lowest pair is: " << lowest_pair[0] << ", " << lowest_pair[1] << "\n";
        if (lowest_pair == std::vector<int>{0,0}) break;

        symbols[lowest_pair[0]] += symbols[lowest_pair[1]];

        symbols.erase(symbols.begin() + lowest_pair[1]);

    }

}

std::vector<int> Tokenizer::encode(const std::string& query) const {
    std::vector<std::string> pre_tokenized = pre_tokenize(query);
    std::map<size_t,size_t> bytes_to_unicode_map = bytes_to_unicode();

    std::vector<std::string> splited_tokens;

    for (std::string& pre_token : pre_tokenized) {
        std::vector<std::string> new_token;
        for (int i = 0; i < pre_token.size(); ++i) {
            new_token.push_back(to_utf8(bytes_to_unicode_map[static_cast<size_t>(pre_token[i])]));
        }

        bpe_merge(new_token);
        for (auto sym : new_token) splited_tokens.push_back(sym);
    }

    std::vector<int> tokens;
    for (auto encoded_token : splited_tokens) tokens.push_back(vocab_map.at(encoded_token));
    return tokens;

}

