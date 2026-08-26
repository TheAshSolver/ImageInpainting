#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <regex>
#include <algorithm>
#include <cstdint>

class CLIPTokenizer {
public:
    static constexpr int32_t SOT_TOKEN = 49406; // <|startoftext|>
    static constexpr int32_t EOT_TOKEN = 49407; // <|endoftext|> / PAD
    static constexpr size_t  MAX_LEN   = 77;

private:
    std::unordered_map<std::string, int32_t> encoder;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    std::unordered_map<uint8_t, std::string> byte_to_unicode;
    std::regex pattern;

    // Unicode byte-level mapping used by GPT2/CLIP
    void init_byte_encoder() {
        std::vector<uint32_t> bs;

        for (int i = '!'; i <= '~'; ++i)
            bs.push_back(i);

        for (int i = 0xA1; i <= 0xAC; ++i)
            bs.push_back(i);

        for (int i = 0xAE; i <= 0xFF; ++i)
            bs.push_back(i);

        std::vector<uint32_t> cs = bs;

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        auto to_utf8 = [](uint32_t cp) -> std::string {
            std::string out;

            if (cp <= 0x7F) {
                out += static_cast<char>(cp);
            } else if (cp <= 0x7FF) {
                out += static_cast<char>(0xC0 | (cp >> 6));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                out += static_cast<char>(0xE0 | (cp >> 12));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }

            return out;
        };

        for (size_t i = 0; i < bs.size(); ++i) {
            byte_to_unicode[static_cast<uint8_t>(bs[i])] =
                to_utf8(cs[i]);
        }
    }

    // Clean whitespace
    std::string whitespace_clean(const std::string& text) {
        std::string res = std::regex_replace(text, std::regex(R"(\s+)"), " ");
        if (!res.empty() && res.front() == ' ') res.erase(res.begin());
        if (!res.empty() && res.back() == ' ') res.pop_back();
        return res;
    }

    // Lowercase conversion
    std::string to_lower(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
        return str;
    }

    // Unescape common JSON escaped sequences in tokens
    std::string unescape_json_string(const std::string& input) {
        std::string out;
        out.reserve(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] == '\\' && i + 1 < input.size()) {
                char next = input[i + 1];
                if (next == '"' || next == '\\' || next == '/') {
                    out += next;
                    ++i;
                    continue;
                }
            }
            out += input[i];
        }
        return out;
    }

    // BPE merge procedure
    std::vector<std::string> bpe(const std::string& token) {
        if (token.empty()) return {};

        std::vector<std::string> word;
        for (size_t i = 0; i < token.size(); ) {
            unsigned char c = token[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            word.push_back(token.substr(i, len));
            i += len;
        }

        if (!word.empty()) word.back() += "</w>";

        while (word.size() > 1) {
            int min_rank = 1e9;
            std::pair<std::string, std::string> best_pair;
            bool found = false;

            for (size_t i = 0; i < word.size() - 1; ++i) {
                std::pair<std::string, std::string> p = {word[i], word[i + 1]};
                auto it = bpe_ranks.find(p);
                if (it != bpe_ranks.end() && it->second < min_rank) {
                    min_rank = it->second;
                    best_pair = p;
                    found = true;
                }
            }

            if (!found) break;

            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size(); ) {
                if (i < word.size() - 1 && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                    new_word.push_back(best_pair.first + best_pair.second);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }
            word = new_word;
        }

        return word;
    }

public:
    CLIPTokenizer() {
        init_byte_encoder();
        // Regex splitting pattern used by OpenAI CLIP
        pattern = std::regex(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s\w]+|\w+|\s+)");
    }

    bool load(const std::string& vocab_path, const std::string& merges_path) {
        // 1. Buffer-wide JSON vocab parser (handles minified single-line and multiline files)
        std::ifstream vf(vocab_path);
        if (!vf.is_open()) {
            std::cerr << "Cannot open vocab file: " << vocab_path << std::endl;
            return false;
        }

        std::stringstream buffer;
        buffer << vf.rdbuf();
        std::string content = buffer.str();

        static const std::regex kv_pattern(R"json("((?:[^"\\]|\\.)*)"\s*:\s*(\d+))json");
        auto begin = std::sregex_iterator(content.begin(), content.end(), kv_pattern);
        auto end = std::sregex_iterator();

        encoder.clear();
        for (auto it = begin; it != end; ++it) {
            std::string key = unescape_json_string((*it)[1].str());
            int32_t id = std::stoi((*it)[2].str());
            encoder[key] = id;
        }

        std::cout << "[CLIPTokenizer] Loaded " << encoder.size() << " vocab entries from " << vocab_path << std::endl;
        if (encoder.empty()) {
            std::cerr << "Warning: No vocabulary entries found in " << vocab_path << std::endl;
            return false;
        }

        // 2. Load merges.txt
        std::ifstream mf(merges_path);
        if (!mf.is_open()) {
            std::cerr << "Cannot open merges file: " << merges_path << std::endl;
            return false;
        }

        bpe_ranks.clear();
        std::string line;
        int rank = 0;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string first, second;
            if (ss >> first >> second) {
                bpe_ranks[{first, second}] = rank++;
            }
        }

        std::cout << "[CLIPTokenizer] Loaded " << bpe_ranks.size() << " BPE merge rules from " << merges_path << std::endl;
        return true;
    }

    std::vector<int32_t> encode(const std::string& text) {
        std::vector<int32_t> tokens(MAX_LEN, EOT_TOKEN);
        tokens[0] = SOT_TOKEN;
        size_t token_idx = 1;

        std::string clean = to_lower(whitespace_clean(text));
        auto words_begin = std::sregex_iterator(clean.begin(), clean.end(), pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::string match = it->str();
            if (match == " ") continue;

            // Map each byte to unicode mapped character
            std::string byte_encoded;
            for (unsigned char b : match) {
                byte_encoded += byte_to_unicode[b];
            }

            // Perform BPE subword segmentation
            std::vector<std::string> subwords = bpe(byte_encoded);
            for (const auto& sub : subwords) {
                if (token_idx >= MAX_LEN - 1) break; // Reserve slot for trailing EOT
                auto enc_it = encoder.find(sub);
                if (enc_it != encoder.end()) {
                    tokens[token_idx++] = enc_it->second;
                }
            }
            if (token_idx >= MAX_LEN - 1) break;
        }

        tokens[token_idx] = EOT_TOKEN;
        return tokens;
    }
};