#!/usr/bin/env python3
"""tok_parity.py VOCAB.json MERGES.txt [--cli ./tok_cli]

Compares the SD runner's C++ tokenizer (tools/tok_cli) with Hugging Face's CLIPTokenizer (the reference SD 1.5 uses)
on prompts that matter for this app: plain words, digits, punctuation, contractions and the Smart-SD photo vocabulary.
Needs:  pip install transformers   (only the tokenizer files vocab.json / merges.txt are used, no model download).
"""
import argparse, subprocess, sys

PROMPTS = [
    "empty waters",
    "photograph of a beautiful empty scene, highest quality settings",
    "calm blue water surface with gentle ripples, smooth, seamless, no objects, soft diffused light, even lighting",
    "35mm lens, f/5.6, crisp detail",
    "shot at f/8, deep depth of field",
    "it's a boat, isn't it?",
    "lush green foliage and grass, rich detailed texture",
    "2024 sunset over the sea, 4k photo",
    "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    "",
]


def main():
    p = argparse.ArgumentParser(); p.add_argument("vocab"); p.add_argument("merges"); p.add_argument("--cli", default="./tok_cli")
    a = p.parse_args()
    from transformers import CLIPTokenizer
    ref = CLIPTokenizer(a.vocab, a.merges)
    bad = 0
    for text in PROMPTS:
        want = ref(text, padding="max_length", max_length=77, truncation=True)["input_ids"]
        out = subprocess.run([a.cli, a.vocab, a.merges, text], capture_output=True, text=True).stdout.split()
        got = [int(v) for v in out]
        ok = got == want
        bad += 0 if ok else 1
        print(("ok   " if ok else "FAIL ") + repr(text))
        if not ok:
            n = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), None)
            print(f"      first difference at index {n}: cli={got[:12]}  hf={want[:12]}")
    print("TOKENIZER PARITY OK" if bad == 0 else f"{bad} prompt(s) tokenise differently")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
