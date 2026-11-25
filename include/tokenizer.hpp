#pragma once

#include "utils.hpp"


class Tokenizer {
    public:
        const int EOS;
        std::vector<int> encode(const std::string&) const;
        std::string decode(const std::vector<int>&) const;
        Tokenizer(const std::string& path);
        std::map<size_t,size_t> bytes_to_unicode() const;
    

        void bpe_merge(std::vector<std::string>&) const;
    private:
        int get_eos_id(std::string);
        std::map<std::string, int> vocab_map;
        std::map<std::vector<std::string>,int> merge_map;
        void load_merge(const std::string&);
        void load_vocab(const std::string&);
        std::vector<std::string> pre_tokenize(const std::string&) const;
        std::vector<int> get_lowest_pair(std::vector<std::string>&) const;
        int get_rank(std::vector<std::string>&) const;




};

