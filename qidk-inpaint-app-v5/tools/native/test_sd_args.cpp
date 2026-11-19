// g++ -std=c++17 test_sd_args.cpp -o test_sd_args && ./test_sd_args
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "sd_args.hpp"

static int fails = 0;
static SdArgs run(std::initializer_list<const char*> args) {
    std::vector<std::string> store(args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : store) argv.push_back(const_cast<char*>(s.c_str()));
    return parseSdArgs(static_cast<int>(argv.size()), argv.data());
}
#define CHECK(c, msg) do { if (c) std::printf("ok   %s\n", msg); else { ++fails; std::printf("FAIL %s\n", msg); } } while (0)

int main() {
    SdArgs a = run({"runner"});
    CHECK(a.ok() && a.steps == 12 && a.cfg == 2.0f && a.seed == 42 && !a.prompt.empty(), "defaults");
    a = run({"runner", "empty waters"});
    CHECK(a.ok() && a.prompt == "empty waters" && a.steps == 12, "legacy: prompt only");
    a = run({"runner", "empty waters", "20"});
    CHECK(a.ok() && a.steps == 20, "legacy: prompt + steps");
    a = run({"runner", "empty waters", ""});
    CHECK(!a.ok(), "legacy: EMPTY argv[2] (the old negative-prompt slot) is rejected, not atoi()'d to 0");
    a = run({"runner", "empty waters", "ugly, blurry"});
    CHECK(!a.ok() && a.error.find("step count") != std::string::npos, "legacy: a negative prompt in argv[2] is rejected with a hint");
    a = run({"runner", "p", "1"});
    CHECK(!a.ok(), "legacy: steps below 2 rejected");
    a = run({"runner", "--prompt", "empty waters, calm sea", "--neg", "boat, ship", "--steps", "20", "--cfg", "7.5", "--seed", "123"});
    CHECK(a.ok() && a.prompt == "empty waters, calm sea" && a.negative == "boat, ship" && a.steps == 20 && a.cfg == 7.5f && a.seed == 123u, "flags: everything");
    a = run({"runner", "--steps", "30", "--prompt", ""});
    CHECK(a.ok() && a.prompt.empty() && a.steps == 30, "flags: empty prompt is allowed and kept (default is not substituted)");
    a = run({"runner", "--steps", "abc"});
    CHECK(!a.ok(), "flags: non-numeric steps rejected");
    a = run({"runner", "--cfg", "0.5"});
    CHECK(!a.ok(), "flags: cfg below 1 rejected");
    a = run({"runner", "--bogus", "1"});
    CHECK(!a.ok(), "flags: unknown option rejected");
    a = run({"runner", "--prompt"});
    CHECK(!a.ok(), "flags: missing value rejected");
    std::printf(fails ? "\n%d FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
