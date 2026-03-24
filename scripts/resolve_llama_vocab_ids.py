#!/usr/bin/env python3
"""
Resolve human-readable words into LLaMA tokenizer token ids.

Why:
- plot_prototype_word_heatmaps_business.py uses fixed (vocab_id, label) pairs.
- LLaMA is a subword tokenizer: a "word" may map to multiple tokens.

This script prints:
- token pieces for each input word/phrase
- ids for each piece
- whether it's a single token (good for clean heatmap x-axis)

Usage examples:
  python scripts/resolve_llama_vocab_ids.py --words "time,period,daily,minute,schedule,season,delay"
  python scripts/resolve_llama_vocab_ids.py --words "volume,zero,empty,peak,load,capacity,shortage"
  python scripts/resolve_llama_vocab_ids.py --words "mine,train,freight,station,route,dispatch,cargo"
  python scripts/resolve_llama_vocab_ids.py --model_id "huggyllama/llama-7b"
"""

import argparse


def get_parser():
    p = argparse.ArgumentParser(description="Resolve words to LLaMA vocab ids")
    p.add_argument("--words", type=str, required=True, help='comma-separated words/phrases, e.g. "time,period,daily"')
    p.add_argument("--model_id", type=str, default="huggyllama/llama-7b", help="HF model id for tokenizer")
    p.add_argument("--prefer_leading_space", action="store_true",
                   help="Also try encoding with a leading space to get ▁word-style tokens")
    return p


def main():
    args = get_parser().parse_args()
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    if not words:
        raise SystemExit("No words provided.")

    try:
        from transformers import LlamaTokenizer
    except Exception as e:
        raise SystemExit("Missing dependency: transformers. Error: {}".format(e))

    tok = None
    for local_only in (True, False):
        try:
            tok = LlamaTokenizer.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=local_only)
            break
        except Exception:
            tok = None
    if tok is None:
        raise SystemExit("Failed to load tokenizer for {}".format(args.model_id))

    def encode(text: str):
        ids = tok.encode(text, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        return ids, pieces

    print("Tokenizer:", args.model_id)
    print("-" * 80)
    for w in words:
        ids1, pieces1 = encode(w)
        out = {
            "input": w,
            "n_pieces": len(ids1),
            "ids": ids1,
            "pieces": pieces1,
        }
        if args.prefer_leading_space:
            ids2, pieces2 = encode(" " + w)
            out["leading_space_ids"] = ids2
            out["leading_space_pieces"] = pieces2
        single = (len(ids1) == 1)
        print("WORD:", repr(w))
        print("  pieces:", pieces1)
        print("  ids   :", ids1, "  (single_token={})".format(single))
        if args.prefer_leading_space:
            print("  [leading space] pieces:", out["leading_space_pieces"])
            print("  [leading space] ids   :", out["leading_space_ids"], " (single_token={})".format(len(out["leading_space_ids"]) == 1))
        print("-" * 80)


if __name__ == "__main__":
    main()

