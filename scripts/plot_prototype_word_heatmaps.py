#!/usr/bin/env python3
"""
绘制「文本原型 × 词集」热力图，类似论文中 learned text prototypes 与 Word Sets 的可视化。
每个子图：若干原型（纵轴）× 某词集内的词（横轴），颜色表示权重强度。

用法：
  cd /home/hesong/Time-LLM
  python scripts/plot_prototype_word_heatmaps.py --export_dir checkpoints/.../prototype_export [--out fig.png]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def get_parser():
    p = argparse.ArgumentParser(description='Plot prototype × word set heatmaps')
    p.add_argument('--export_dir', type=str, required=True, help='prototype_export 目录（含 prototype_word_weight_matrix.npy 与 prototype_topk_words.csv）')
    p.add_argument('--out', type=str, default='', help='输出图片路径，默认 export_dir/prototype_word_sets_heatmaps.png')
    p.add_argument('--num_prototypes', type=int, default=10, help='每个热力图展示的原型数量（纵轴）')
    p.add_argument('--max_words_per_set', type=int, default=13, help='每个词集最多展示的词数（横轴）')
    p.add_argument('--dpi', type=int, default=150)
    return p


def load_data(export_dir):
    W_path = os.path.join(export_dir, 'prototype_word_weight_matrix.npy')
    csv_path = os.path.join(export_dir, 'prototype_topk_words.csv')
    if not os.path.isfile(W_path) or not os.path.isfile(csv_path):
        raise FileNotFoundError('需要 prototype_word_weight_matrix.npy 与 prototype_topk_words.csv')
    W = np.load(W_path)
    df = pd.read_csv(csv_path)
    return W, df


def build_vocab_id_to_word(df):
    """vocab_id -> word，去重取首次出现。"""
    out = {}
    for _, r in df.iterrows():
        vid = int(r['vocab_id'])
        if vid not in out:
            out[vid] = str(r['word']).strip()
    return out


def define_word_sets(df, id2word, max_per_set=13):
    """
    按语义定义 3 个词集，从 CSV 中出现的词里选。
    Set 1: 时间/周期 (temporal)
    Set 2: 数值/统计 (quantity, statistics)
    Set 3: 领域/一般 (domain, e.g. volume, mine, produces)
    """
    # 收集 CSV 中所有 (vocab_id, word)，去重并保留顺序
    seen = set()
    vocab_list = []
    for _, r in df.iterrows():
        vid = int(r['vocab_id'])
        if vid not in seen:
            seen.add(vid)
            vocab_list.append(vid)
    word_lower = {vid: id2word.get(vid, '').lower() for vid in vocab_list}

    keywords_set1 = ['minute', 'period', 'step', 'time', 'sometimes', 'term', 'day', 'trend', 'season', 'cycle', 'times', 'termine']
    keywords_set2 = ['volume', 'value', 'average', 'number', 'stats', 'stat', 'system', 'addition', 'sales', 'quantile', 'max', 'min', 'offset']
    keywords_set3 = ['mine', 'produces', 'credit', 'line', 'data', 'transport', 'underlying', 'conv', 'status', 'perform', 'delete', 'underlying']

    def match_set(vid, keywords):
        w = word_lower.get(vid, '')
        for k in keywords:
            if k in w or w in k:
                return True
        return False

    set1 = [vid for vid in vocab_list if match_set(vid, keywords_set1)][:max_per_set]
    set2 = [vid for vid in vocab_list if match_set(vid, keywords_set2)][:max_per_set]
    set3 = [vid for vid in vocab_list if match_set(vid, keywords_set3)][:max_per_set]

    # 不足时从与任务相关的原型(43,44,45,74)的 top 词中按主题补足
    for pid in [43, 44, 45, 74]:
        for _, r in df[df['prototype_id'] == pid].iterrows():
            vid = int(r['vocab_id'])
            w = word_lower.get(vid, '')
            if len(set1) < max_per_set and vid not in set1 and match_set(vid, keywords_set1):
                set1.append(vid)
            if len(set2) < max_per_set and vid not in set2 and match_set(vid, keywords_set2):
                set2.append(vid)
            if len(set3) < max_per_set and vid not in set3 and match_set(vid, keywords_set3):
                set3.append(vid)
    set1, set2, set3 = set1[:max_per_set], set2[:max_per_set], set3[:max_per_set]

    return [
        (set1, 'Word Set 1 (temporal)', ['minute', 'periodo', 'Step', 'times', '...']),
        (set2, 'Word Set 2 (quantity/stat)', ['volume', 'addition', 'sales', 'stats', '...']),
        (set3, 'Word Set 3 (domain/general)', ['mine', 'produces', 'credit', 'line', '...']),
    ]


def plot_heatmaps(W, df, export_dir, out_path, num_prototypes=10, max_words=13, dpi=150):
    id2word = build_vocab_id_to_word(df)
    word_sets = define_word_sets(df, id2word, max_per_set=max_words)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print('请安装 matplotlib 与 seaborn: pip install matplotlib seaborn')
        return None

    n_sets = len(word_sets)
    fig, axes = plt.subplots(1, n_sets, figsize=(5 * n_sets, 4))
    if n_sets == 1:
        axes = [axes]

    for ax, (vocab_ids, title, example_words) in zip(axes, word_sets):
        if not vocab_ids:
            ax.text(0.5, 0.5, 'No words', ha='center', va='center')
            ax.set_title(title)
            continue
        # 子矩阵: 前 num_prototypes 个原型 × 该词集的词
        protos = np.arange(min(num_prototypes, W.shape[0]))
        W_sub = W[protos][:, vocab_ids]
        words = [id2word.get(vid, str(vid)) for vid in vocab_ids]
        # 短标签，避免重叠
        words_short = [w.replace('▁', '')[:10] for w in words]
        sns.heatmap(
            W_sub,
            xticklabels=words_short,
            yticklabels=[str(i) for i in protos],
            ax=ax,
            cmap='RdYlBu_r',
            center=0,
            cbar_kws={'label': 'weight'},
            annot=False,
        )
        ax.set_xlabel('Word')
        ax.set_ylabel('Prototype')
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    fig.suptitle('Visualization of {} learned text prototypes × Word Sets'.format(num_prototypes), fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print('已保存:', out_path)
    return out_path


def main():
    args = get_parser().parse_args()
    export_dir = args.export_dir.rstrip('/')
    out_path = args.out or os.path.join(export_dir, 'prototype_word_sets_heatmaps.png')

    W, df = load_data(export_dir)
    plot_heatmaps(
        W, df, export_dir, out_path,
        num_prototypes=args.num_prototypes,
        max_words=args.max_words_per_set,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
