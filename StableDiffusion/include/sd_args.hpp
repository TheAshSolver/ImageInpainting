// sd_args.hpp - command-line parsing for sd_runner_inpaint (header only, no dependencies).
//
//   new form : runner --prompt "<text>" --neg "<text>" --steps 20 --cfg 2.0 --seed 42     (any subset, any order)
//   legacy   : runner "<prompt>" [<steps>]       argv[2] is the STEP COUNT. A non-numeric argv[2] is an ERROR.
//
// Why: the Android driver used to call `runner "<prompt>" "<negative prompt>"` while the runner did
// `numSteps = atoi(argv[2])`, so every request ran with 0 (clamped to 2) denoising steps and the prompt had no effect.
// Bad input now stops the run with a message instead of silently becoming a number.
#pragma once
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

struct SdArgs {
    std::string prompt   = "photograph of a beautiful empty scene, highest quality settings";
    std::string negative = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured";
    int steps = 12;
    float cfg = 2.0f;
    unsigned seed = 42;
    std::string error;                     // non-empty => do not run; print it and exit(2)
    bool ok() const { return error.empty(); }
};

namespace sdargs_detail {
inline bool toInt(const char* s, long lo, long hi, long& out) {
    if (!s || !*s) return false;
    char* end = nullptr; errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < lo || v > hi) return false;
    out = v; return true;
}
inline bool toFloat(const char* s, float lo, float hi, float& out) {
    if (!s || !*s) return false;
    char* end = nullptr; errno = 0;
    float v = std::strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !(v >= lo && v <= hi)) return false;
    out = v; return true;
}
}  // namespace sdargs_detail

inline SdArgs parseSdArgs(int argc, char** argv) {
    using namespace sdargs_detail;
    SdArgs a;
    bool flagMode = false;
    for (int i = 1; i < argc; ++i) if (std::strncmp(argv[i], "--", 2) == 0) flagMode = true;

    if (!flagMode) {
        if (argc > 1) a.prompt = argv[1];
        if (argc > 2) {
            long v;
            if (!toInt(argv[2], 2, 100, v)) {
                a.error = std::string("argv[2] must be the step count (2..100), got '") + argv[2] +
                          "'. Use --neg \"...\" for a negative prompt.";
                return a;
            }
            a.steps = static_cast<int>(v);
        }
        if (argc > 3) a.error = "too many positional arguments; use --prompt/--neg/--steps/--cfg/--seed";
        return a;
    }
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        if (std::strncmp(k, "--", 2) != 0) { a.error = std::string("unexpected argument '") + k + "'"; return a; }
        if (i + 1 >= argc) { a.error = std::string("missing value after ") + k; return a; }
        const char* v = argv[++i];
        long n; float f;
        if (!std::strcmp(k, "--prompt")) a.prompt = v;
        else if (!std::strcmp(k, "--neg")) a.negative = v;
        else if (!std::strcmp(k, "--steps")) { if (!toInt(v, 2, 100, n)) { a.error = std::string("--steps must be 2..100, got '") + v + "'"; return a; } a.steps = static_cast<int>(n); }
        else if (!std::strcmp(k, "--cfg")) { if (!toFloat(v, 1.0f, 20.0f, f)) { a.error = std::string("--cfg must be 1..20, got '") + v + "'"; return a; } a.cfg = f; }
        else if (!std::strcmp(k, "--seed")) { if (!toInt(v, 0, 2147483647L, n)) { a.error = std::string("--seed must be 0..2147483647, got '") + v + "'"; return a; } a.seed = static_cast<unsigned>(n); }
        else { a.error = std::string("unknown option ") + k; return a; }
    }
    return a;
}
