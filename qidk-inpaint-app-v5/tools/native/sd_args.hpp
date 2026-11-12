#pragma once

#include <string>
#include <vector>
#include <cstdlib>
#include <cctype>
#include <sstream>

struct SdArgs {
    std::string prompt = "a photorealistic photograph, fine detail, masterwork";
    std::string negative = "";
    int steps = 12;
    float cfg = 2.0f;
    unsigned int seed = 42;
    std::string error = "";

    bool ok() const { return error.empty(); }
};

inline bool isIntegerString(const std::string& s) {
    if (s.empty()) return false;
    size_t start = 0;
    if (s[0] == '-' || s[0] == '+') {
        if (s.size() == 1) return false;
        start = 1;
    }
    for (size_t i = start; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

inline bool isFloatString(const std::string& s, float& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtof(s.c_str(), &end);
    return end != nullptr && *end == '\0';
}

inline SdArgs parseSdArgs(int argc, char** argv) {
    SdArgs args;
    if (argc <= 1) {
        return args;
    }

    std::string first = argv[1];
    // Check if positional (does not start with '--')
    if (first.rfind("--", 0) != 0) {
        args.prompt = first;
        if (argc >= 3) {
            std::string second = argv[2];
            if (second.empty()) {
                args.error = "legacy: EMPTY argv[2] (the old negative-prompt slot) is rejected, not atoi()'d to 0";
                return args;
            }
            if (!isIntegerString(second)) {
                args.error = "legacy: invalid step count in argv[2]: '" + second + "' (negative prompt in argv[2] is rejected with a hint)";
                return args;
            }
            int s = std::atoi(second.c_str());
            if (s < 2) {
                args.error = "legacy: steps below 2 rejected";
                return args;
            }
            args.steps = s;
        }
        return args;
    }

    // Flag-based parsing
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        if (opt == "--prompt") {
            if (i + 1 >= argc) {
                args.error = "flags: missing value for --prompt";
                return args;
            }
            args.prompt = argv[++i];
        } else if (opt == "--neg" || opt == "--negative") {
            if (i + 1 >= argc) {
                args.error = "flags: missing value for --neg";
                return args;
            }
            args.negative = argv[++i];
        } else if (opt == "--steps") {
            if (i + 1 >= argc) {
                args.error = "flags: missing value for --steps";
                return args;
            }
            std::string val = argv[++i];
            if (!isIntegerString(val)) {
                args.error = "flags: non-numeric steps rejected";
                return args;
            }
            int s = std::atoi(val.c_str());
            if (s < 2) {
                args.error = "flags: steps below 2 rejected";
                return args;
            }
            args.steps = s;
        } else if (opt == "--cfg") {
            if (i + 1 >= argc) {
                args.error = "flags: missing value for --cfg";
                return args;
            }
            std::string val = argv[++i];
            float c = 0.0f;
            if (!isFloatString(val, c)) {
                args.error = "flags: non-numeric cfg rejected";
                return args;
            }
            if (c < 1.0f) {
                args.error = "flags: cfg below 1 rejected";
                return args;
            }
            args.cfg = c;
        } else if (opt == "--seed") {
            if (i + 1 >= argc) {
                args.error = "flags: missing value for --seed";
                return args;
            }
            std::string val = argv[++i];
            if (!isIntegerString(val)) {
                args.error = "flags: non-numeric seed rejected";
                return args;
            }
            args.seed = static_cast<unsigned int>(std::strtoul(val.c_str(), nullptr, 10));
        } else {
            args.error = "flags: unknown option rejected: " + opt;
            return args;
        }
    }

    return args;
}
