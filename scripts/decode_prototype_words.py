#!/usr/bin/env python3
"""
把 prototype_topk_words.json 里的 vocab_id 解码成 LLaMA 的真实 token 字符串，
并可选只保留前 N 个原型（如 100），便于查看「具体是哪些词」。

依赖：能加载 LLaMA tokenizer（transformers + sentencepiece）。若无法加载则跳过解码，仅做截断。

用法：
  cd /home/hesong/Time-LLM
  python scripts/decode_prototype_words.py --export_dir checkpoints/.../prototype_export [--max_prototypes 100]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def get_parser():
    p = argparse.ArgumentParser(description='Decode vocab_id to real token strings for prototype top-k words')
    p.add_argument('--export_dir', type=str, required=True, help='prototype_export 目录（含 prototype_topk_words.json）')
    p.add_argument('--max_prototypes', type=int, default=0, help='只输出前 N 个原型（0=全部）')
    p.add_argument('--out', type=str, default='', help='输出 JSON 路径，默认 export_dir/prototype_topk_words_decoded.json')
    return p


def load_tokenizer():
    try:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True,
        )
        return tokenizer
    except Exception:
        try:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
            )
            return tokenizer
        except Exception as e:
            print('无法加载 LLaMA tokenizer:', e)
            return None


def main():
    args = get_parser().parse_args()
    export_dir = args.export_dir
    in_path = os.path.join(export_dir, 'prototype_topk_words.json')
    if not os.path.isfile(in_path):
        print('未找到:', in_path)
        return
    out_path = args.out or os.path.join(export_dir, 'prototype_topk_words_decoded.json')

    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 只保留前 max_prototypes 个原型
    if args.max_prototypes > 0:
        keys = sorted([int(k) for k in data.keys()])
        keys = [k for k in keys if k < args.max_prototypes]
        data = {str(k): data[str(k)] for k in keys}
        print('只保留前 {} 个原型，共 {} 条词'.format(args.max_prototypes, sum(len(v) for v in data.values())))

    tokenizer = load_tokenizer()
    if tokenizer is None:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print('未解码词面，已按原样写出（可限制原型数）:', out_path)
        return

    # 收集所有出现的 vocab_id 并解码为 token 字符串
    vocab_ids = set()
    for protos in data.values():
        for item in protos:
            vocab_ids.add(item.get('vocab_id'))
    id_to_word = {}
    for vid in vocab_ids:
        try:
            # 用 convert_ids_to_tokens 得到原始 token 字符串（如 "▁the"）
            tok = tokenizer.convert_ids_to_tokens([int(vid)])
            id_to_word[vid] = tok[0] if tok else ('[id_{}]'.format(vid))
        except Exception:
            id_to_word[vid] = 'id_{}'.format(vid)

    # 在每条里加上 decoded_word 或替换 word
    decoded = {}
    for pid, items in data.items():
        decoded[pid] = []
        for it in items:
            d = dict(it)
            vid = d.get('vocab_id')
            d['word'] = id_to_word.get(vid, d.get('word', 'id_{}'.format(vid)))
            decoded[pid].append(d)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(decoded, f, ensure_ascii=False, indent=1)
    print('已写出带真实词面的 JSON:', out_path)

    # 同时写一个精简版 CSV：前 100 个原型 × top 词，便于快速看
    csv_path = out_path.replace('.json', '.csv')
    import csv
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['prototype_id', 'rank', 'vocab_id', 'word', 'weight'])
        for pid in sorted(decoded.keys(), key=int):
            for it in decoded[pid]:
                w.writerow([pid, it.get('rank'), it.get('vocab_id'), it.get('word'), it.get('weight')])
    print('已写出 CSV:', csv_path)


if __name__ == '__main__':
    main()
