#!/usr/bin/env python3
"""
从主模型 checkpoint 提取重编程层（Reprogramming Layer）的文本原型（Text Prototypes）细节，
用于可解释性分析与作图。

输出：
- 文本原型数量：num_tokens = 1000（learnable prototypes）
- 原始词表大小：vocab_size（LLaMA 约 32k）
- 权重关系矩阵 W：(num_tokens, vocab_size)，W[i,j] = 原型 i 对词 j 的权重
- 每个原型的 top-k 词及权重
- 可选：热力图、原型-词表关系可视化

用法：
  cd /home/hesong/Time-LLM
  python scripts/extract_reprogramming_prototypes.py --ckpt_dir checkpoints/long_term_forecast_Iron_96_48_TimeLLM_..._0-iron
  或使用默认主模型路径（见下方 default_ckpt_dir）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 默认主模型 checkpoint 目录（与 Eval_Iron.sh 一致）
DEFAULT_CKPT_DIR = os.path.join(
    ROOT, 'checkpoints',
    'long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron'
)


def get_parser():
    p = argparse.ArgumentParser(description='Extract text prototypes and prototype-vocab weight matrix from Time-LLM checkpoint')
    p.add_argument('--ckpt_dir', type=str, default='', help='checkpoint 目录（含 checkpoint 文件）')
    p.add_argument('--top_k', type=int, default=20, help='每个原型保留的 top-k 词')
    p.add_argument('--out_dir', type=str, default='', help='输出目录，默认为 ckpt_dir/prototype_export')
    p.add_argument('--plot', action='store_true', help='是否绘制热力图（需 matplotlib/seaborn）')
    p.add_argument('--plot_top_vocab', type=int, default=200, help='热力图仅显示权重方差最大的前 N 个词')
    p.add_argument('--weights_only', action='store_true', help='仅从 checkpoint 读权重，不加载完整模型（无需 LLaMA/transformers/tokenizer）')
    p.add_argument('--max_prototypes', type=int, default=0, help='只导出前 N 个原型的 top-k 词（0 表示全部 1000 个）')
    p.add_argument('--decode_words', action='store_true', help='尝试加载 LLaMA tokenizer，将 vocab_id 解码为真实 token 字符串（需 transformers+sentencepiece）')
    return p


def load_llama_tokenizer():
    """仅加载 LLaMA tokenizer，不加载模型。用于将 vocab_id 转为真实词。"""
    try:
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(
            'huggyllama/llama-7b', trust_remote_code=True, local_files_only=True)
    except Exception:
        try:
            from transformers import LlamaTokenizer
            return LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b', trust_remote_code=True, local_files_only=False)
        except Exception:
            return None


def load_weights_only(ckpt_dir):
    """仅从 checkpoint 的 state_dict 读取 mapping_layer 与 reprogramming 相关权重，不加载 LLaMA。
    返回 W, num_tokens, vocab_size, extra_state（含 bias 与 reprogramming 层参数）。"""
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(ckpt_path))
    state = torch.load(ckpt_path, map_location='cpu')
    if 'mapping_layer.weight' not in state:
        raise KeyError('checkpoint 中无 mapping_layer.weight，请确认是 Time-LLM 的 checkpoint')
    w_t = state['mapping_layer.weight'].detach().float().cpu()
    W = w_t.numpy()
    num_tokens, vocab_size = W.shape
    # 额外保存的键（用于可解释性/作图）
    extra_keys = [
        'mapping_layer.bias',
        'reprogramming_layer.query_projection.weight', 'reprogramming_layer.query_projection.bias',
        'reprogramming_layer.key_projection.weight', 'reprogramming_layer.key_projection.bias',
        'reprogramming_layer.value_projection.weight', 'reprogramming_layer.value_projection.bias',
        'reprogramming_layer.out_projection.weight', 'reprogramming_layer.out_projection.bias',
    ]
    extra_state = {}
    for k in extra_keys:
        if k in state:
            t = state[k].detach().float().cpu().numpy()
            extra_state[k] = t
    return W, num_tokens, vocab_size, extra_state


def load_model_and_ckpt(ckpt_dir):
    """加载 Time-LLM 模型并加载 checkpoint；返回 model 和 tokenizer。"""
    from utils.tools import load_content
    from models import TimeLLM

    ckpt_path = os.path.join(ckpt_dir, 'checkpoint')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(ckpt_path))

    # 使用与 Eval_Iron 一致的默认参数构建模型
    class Args:
        task_name = 'long_term_forecast'
        model_id = 'Iron_96_48'
        model = 'TimeLLM'
        data = 'custom'
        features = 'M'
        seq_len = 96
        label_len = 48
        pred_len = 48
        enc_in = 4
        dec_in = 4
        c_out = 4
        d_model = 32
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 128
        factor = 3
        embed = 'timeF'
        des = 'Iron_Ore_Transport_Exp'
        patch_len = 16
        stride = 8
        dropout = 0.1
        llm_model = 'LLAMA'
        llm_dim = 4096
        llm_layers = 32
        prompt_type = 'full'
        model_comment = 'iron'

    args = Args()
    args.content = load_content(args)
    model = TimeLLM.Model(args).float()
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    tokenizer = model.tokenizer
    return model, tokenizer


def extract_and_save(ckpt_dir, out_dir, top_k=20, do_plot=False, plot_top_vocab=200, weights_only=False, max_prototypes=0, decode_words=False):
    if weights_only:
        W, num_tokens, vocab_size, extra_state = load_weights_only(ckpt_dir)
        word_embeddings = None
        id_to_word = None  # 无 tokenizer，仅用 vocab_id
    else:
        try:
            model, tokenizer = load_model_and_ckpt(ckpt_dir)
        except Exception as e:
            print('完整模型加载失败 ({})，改用 --weights_only 仅提取权重。'.format(e))
            weights_only = True
            W, num_tokens, vocab_size, extra_state = load_weights_only(ckpt_dir)
            word_embeddings = None
            id_to_word = None
        else:
            num_tokens = model.num_tokens
            vocab_size = model.vocab_size
            W = model.mapping_layer.weight.detach().float().cpu().numpy()
            word_embeddings = model.word_embeddings.detach().float().cpu().numpy()
            # 一并提取重编程层与 mapping bias，便于可解释性
            extra_state = {
                'mapping_layer.bias': model.mapping_layer.bias.detach().float().cpu().numpy(),
                'reprogramming_layer.query_projection.weight': model.reprogramming_layer.query_projection.weight.detach().float().cpu().numpy(),
                'reprogramming_layer.query_projection.bias': model.reprogramming_layer.query_projection.bias.detach().float().cpu().numpy(),
                'reprogramming_layer.key_projection.weight': model.reprogramming_layer.key_projection.weight.detach().float().cpu().numpy(),
                'reprogramming_layer.key_projection.bias': model.reprogramming_layer.key_projection.bias.detach().float().cpu().numpy(),
                'reprogramming_layer.value_projection.weight': model.reprogramming_layer.value_projection.weight.detach().float().cpu().numpy(),
                'reprogramming_layer.value_projection.bias': model.reprogramming_layer.value_projection.bias.detach().float().cpu().numpy(),
                'reprogramming_layer.out_projection.weight': model.reprogramming_layer.out_projection.weight.detach().float().cpu().numpy(),
                'reprogramming_layer.out_projection.bias': model.reprogramming_layer.out_projection.bias.detach().float().cpu().numpy(),
            }
            try:
                id_to_word = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
            except Exception:
                id_to_word = ['id_{}'.format(i) for i in range(vocab_size)]

    os.makedirs(out_dir, exist_ok=True)

    # 1) 元信息
    meta = {
        'num_tokens': int(num_tokens),
        'vocab_size': int(vocab_size),
        'weight_matrix_shape': [int(num_tokens), int(vocab_size)],
        'weights_only': weights_only,
        'description': 'W[i,j] = 文本原型 i 对词 j 的线性权重；source_embeddings = W @ word_embeddings 得到 1000 个原型在 LLM 空间的向量',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print('num_tokens (文本原型数):', num_tokens)
    print('vocab_size (原始词表大小):', vocab_size)
    print('权重矩阵 W 形状:', W.shape)

    # 2) 完整权重矩阵
    np.save(os.path.join(out_dir, 'prototype_word_weight_matrix.npy'), W)
    print('已保存: prototype_word_weight_matrix.npy')

    # 2a) 重编程层与 mapping 的其余参数（weights_only 时从 checkpoint 读出）
    if extra_state:
        for key, arr in extra_state.items():
            name = key.replace('.', '_') + '.npy'
            np.save(os.path.join(out_dir, name), arr)
        print('已保存: mapping_layer_bias, reprogramming_layer 的 query/key/value/out 投影权重与偏置')

    if word_embeddings is not None:
        source_emb = (W @ word_embeddings).astype(np.float32)
        np.save(os.path.join(out_dir, 'prototype_embeddings_llm_space.npy'), source_emb)
        meta['prototype_embeddings_shape'] = list(source_emb.shape)
        with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print('已保存: prototype_embeddings_llm_space.npy (1000 个原型在 LLM 嵌入空间的向量)')
    else:
        print('(weights_only 模式未保存 prototype_embeddings_llm_space.npy，需完整模型)')

    # 3) 每个原型的 top-k 词及权重（无 tokenizer 时 word 列为 vocab_id 字符串）
    if id_to_word is None:
        id_to_word = ['id_{}'.format(i) for i in range(vocab_size)]

    n_protos = min(num_tokens, max_prototypes) if max_prototypes > 0 else num_tokens
    rows = []
    for i in range(n_protos):
        row_weights = W[i, :]
        top_indices = np.argsort(-np.abs(row_weights))[:top_k]
        for r, idx in enumerate(top_indices):
            idx = int(idx)
            rows.append({
                'prototype_id': i,
                'rank': r + 1,
                'vocab_id': idx,
                'word': id_to_word[idx] if idx < len(id_to_word) else 'id_{}'.format(idx),
                'weight': float(row_weights[idx]),
            })
    if max_prototypes > 0:
        print('仅导出前 {} 个原型的 top-{} 词（共 {} 条）'.format(n_protos, top_k, len(rows)))

    # 若指定 --decode_words 且当前为 id_xxx 占位，尝试仅加载 tokenizer 将 vocab_id 转为真实 token
    if decode_words and weights_only:
        tokenizer = load_llama_tokenizer()
        if tokenizer is not None:
            vocab_ids_in_rows = set(r['vocab_id'] for r in rows)
            id_to_word = {}
            for vid in vocab_ids_in_rows:
                try:
                    tok = tokenizer.convert_ids_to_tokens([int(vid)])
                    id_to_word[vid] = tok[0] if tok else 'id_{}'.format(vid)
                except Exception:
                    id_to_word[vid] = 'id_{}'.format(vid)
            for r in rows:
                r['word'] = id_to_word.get(r['vocab_id'], r['word'])
            print('已用 LLaMA tokenizer 将 vocab_id 解码为真实 token 字符串')
        else:
            print('未安装或无法加载 LLaMA tokenizer（需 transformers + sentencepiece），word 仍为 id_xxx；可安装后加 --decode_words 重跑')
    elif weights_only and rows and str(rows[0].get('word', '')).startswith('id_'):
        print('提示：若需将 vocab_id 显示为真实 token 字符串，请安装 sentencepiece 后加 --decode_words 再运行')

    import csv
    csv_path = os.path.join(out_dir, 'prototype_topk_words.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['prototype_id', 'rank', 'vocab_id', 'word', 'weight'])
        w.writeheader()
        w.writerows(rows)
    print('已保存: prototype_topk_words.csv')

    by_prototype = {}
    for r in rows:
        pid = r['prototype_id']
        if pid not in by_prototype:
            by_prototype[pid] = []
        by_prototype[pid].append({'rank': r['rank'], 'word': r['word'], 'vocab_id': r['vocab_id'], 'weight': r['weight']})
    with open(os.path.join(out_dir, 'prototype_topk_words.json'), 'w', encoding='utf-8') as f:
        json.dump(by_prototype, f, ensure_ascii=False, indent=1)
    print('已保存: prototype_topk_words.json')

    # 4) 可选：热力图
    if do_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print('未安装 matplotlib/seaborn，跳过绘图。pip install matplotlib seaborn')
            return out_dir

        var_per_vocab = np.var(W, axis=0)
        top_vocab_idx = np.argsort(-var_per_vocab)[:plot_top_vocab]
        W_sub = W[:, top_vocab_idx]
        words_sub = [id_to_word[i] if i < len(id_to_word) else str(i) for i in top_vocab_idx]

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(W_sub, xticklabels=False, yticklabels=False, ax=ax, cmap='RdBu_r', center=0,
                    cbar_kws={'label': '权重'})
        ax.set_xlabel('词表子集（按权重方差取前 {} 个）'.format(plot_top_vocab))
        ax.set_ylabel('文本原型 (0–999)')
        ax.set_title('重编程层：文本原型 × 词表 权重矩阵（子集）')
        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'prototype_word_heatmap.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print('已保存: prototype_word_heatmap.png')

        np.save(os.path.join(out_dir, 'prototype_word_weight_matrix_sub.npy'), W_sub)
        with open(os.path.join(out_dir, 'heatmap_vocab_subset.txt'), 'w', encoding='utf-8') as f:
            for i, w in enumerate(words_sub):
                f.write('{}\t{}\n'.format(top_vocab_idx[i], w))

    return out_dir


def main():
    parser = get_parser()
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir or DEFAULT_CKPT_DIR
    out_dir = args.out_dir or os.path.join(ckpt_dir, 'prototype_export')

    if not os.path.isdir(ckpt_dir):
        print('Checkpoint 目录不存在:', ckpt_dir)
        print('请指定 --ckpt_dir 或使用默认主模型路径。')
        return
    print('Checkpoint 目录:', ckpt_dir)
    print('输出目录:', out_dir)
    extract_and_save(ckpt_dir, out_dir, top_k=args.top_k, do_plot=args.plot, plot_top_vocab=args.plot_top_vocab, weights_only=args.weights_only, max_prototypes=args.max_prototypes, decode_words=args.decode_words)
    print('完成。可基于 prototype_word_weight_matrix.npy 与 prototype_topk_words.csv 做可解释性分析与作图。')


if __name__ == '__main__':
    main()
