// tok_cli.cpp - prints the 77 CLIP token ids the SD runner would feed the text encoder.
//   g++ -std=c++17 -I "$REPO/StableDiffusion/include" tools/native/tok_cli.cpp -o tok_cli
//   ./tok_cli vocab.json merges.txt "35mm lens, f/8"   ->  49406 ... 49407 49407 ...
#include <cstdio>
#include <iostream>
#include "clip_tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: tok_cli vocab.json merges.txt \"prompt\"\n"); return 2; }
    CLIPTokenizer t;
    if (!t.load(argv[1], argv[2])) return 1;
    for (int32_t id : t.encode(argv[3])) std::printf("%d ", id);
    std::printf("\n");
    return 0;
}
