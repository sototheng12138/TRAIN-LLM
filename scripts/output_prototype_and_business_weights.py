#!/usr/bin/env python3
"""
输出主模型「前文本原型」及与「时序预测 / 铁路业务 / 数据」相关词在权重矩阵中的权重。
与 plot_prototype_word_heatmaps_business.py 使用同一套业务词集，便于对照热力图。

用法：
  python scripts/output_prototype_and_business_weights.py --export_dir checkpoints/.../prototype_export
  python scripts/output_prototype_and_business_weights.py --export_dir checkpoints/.../prototype_export --num_prototypes 50 --out_txt prototype_report.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 与 plot_prototype_word_heatmaps_business.py 一致：时序/预测步、货运量/数值、领域/业务
BUSINESS_WORD_SETS = [
    ('时序与预测步 (Time & horizon)', [
        (11015, 'minute'), (17848, 'periodo'), (14448, 'Step'),
        (3706, 'times'), (18512, 'Sometimes'), (20202, 'termine'),
    ]),
    ('货运量与数值 (Volume & statistics)', [
        (7977, 'volume'), (6124, 'addition'), (16538, 'sales'),
        (1788, 'system'), (16202, 'stats'), (16034, 'Status'), (9210, 'offset'),
    ]),
    ('领域与业务 (Domain & business)', [
        (7903, 'mine'), (13880, 'produces'), (16200, 'credit'),
        (1196, 'line'), (14407, 'underlying'), (7602, 'conv'),
    ]),
]


def main():
    p = argparse.ArgumentParser(description='Output top text prototypes and business word weights')
    p.add_argument('--export_dir', type=str, required=True, help='prototype_export 目录')
    p.add_argument('--num_prototypes', type=int, default=30, help='输出前 N 个文本原型的 top 词')
    p.add_argument('--top_k_words', type=int, default=15, help='每个原型显示的 top-k 词数')
    p.add_argument('--out_txt', type=str, default='', help='输出文本路径，默认 export_dir/prototype_and_business_weights_report.txt')
    args = p.parse_args()

    export_dir = args.export_dir.rstrip('/')
    if not os.path.isdir(export_dir):
        alt = os.path.join(export_dir, 'prototype_export')
        if os.path.isdir(alt):
            export_dir = alt
        else:
            print('未找到目录:', args.export_dir)
            return

    csv_path = os.path.join(export_dir, 'prototype_topk_words.csv')
    W_path = os.path.join(export_dir, 'prototype_word_weight_matrix.npy')
    if not os.path.isfile(csv_path):
        print('未找到:', csv_path)
        return
    if not os.path.isfile(W_path):
        print('未找到:', W_path)
        return

    df = pd.read_csv(csv_path)
    W = np.load(W_path)  # (num_prototypes, vocab_size)
    n_proto, vocab_size = W.shape

    out_path = args.out_txt or os.path.join(export_dir, 'prototype_and_business_weights_report.txt')
    lines = []

    # ----- 1. 前文本原型：每个原型的 top-k 词 -----
    lines.append('=' * 70)
    lines.append('主模型 前 {} 个文本原型（每个原型 top-{} 词）'.format(args.num_prototypes, args.top_k_words))
    lines.append('=' * 70)
    for pid in range(min(args.num_prototypes, int(df['prototype_id'].max()) + 1)):
        sub = df[df['prototype_id'] == pid].head(args.top_k_words)
        words = sub.apply(lambda r: '{} ({:.5f})'.format(r['word'], r['weight']), axis=1).tolist()
        lines.append('原型 {:3d}:  '.format(pid) + '  |  '.join(words))
    lines.append('')

    # ----- 2. 与「时序预测 / 铁路业务 / 数据」相关词的权重 -----
    lines.append('=' * 70)
    lines.append('与 时序预测 / 铁路业务 / 数据 相关词在权重矩阵 W 中的统计（前 {} 个原型）'.format(args.num_prototypes))
    lines.append('  W 形状: {} x {} (原型 x 词表)'.format(n_proto, vocab_size))
    lines.append('=' * 70)
    proto_range = min(args.num_prototypes, n_proto)
    for set_name, word_list in BUSINESS_WORD_SETS:
        lines.append('')
        lines.append('【{}】'.format(set_name))
        vocab_ids = [v for v, _ in word_list]
        labels = [lbl for _, lbl in word_list]
        # 取 W 的前 proto_range 行，对应列取 vocab_ids（需在 vocab_size 内）
        valid_ids = [v for v in vocab_ids if v < vocab_size]
        valid_lbls = [labels[vocab_ids.index(v)] for v in valid_ids]
        if not valid_ids:
            lines.append('  (无有效 vocab_id)')
            continue
        W_sub = W[:proto_range][:, np.array(valid_ids)]  # (proto_range, len(valid_ids))
        lines.append('  词\t\t\t均值(前{}原型)\t最大值\t最大所在原型'.format(proto_range))
        lines.append('  ' + '-' * 60)
        for j, (vid, lbl) in enumerate(zip(valid_ids, valid_lbls)):
            col = W_sub[:, j]
            mean_w = float(np.mean(col))
            max_w = float(np.max(col))
            argmax_p = int(np.argmax(col))
            lines.append('  {:12s}\t{:+.6f}\t{:+.6f}\t{}'.format(lbl[:12], mean_w, max_w, argmax_p))
        # 子矩阵：前 10 个原型 × 该词集（便于和热力图对照）
        lines.append('  前 10 个原型 × 本词集 权重子块（与热力图一致）:')
        block = W[:10][:, np.array(valid_ids)]
        for i in range(10):
            line = '     P{:2d}  '.format(i) + '  '.join(['{:+.4f}'.format(block[i, j]) for j in range(len(valid_ids))])
            lines.append(line)
        lines.append('  列顺序: ' + ', '.join(valid_lbls))
    lines.append('')
    lines.append('=' * 70)
    lines.append('出图可运行: python scripts/plot_prototype_word_heatmaps_business.py --export_dir "{}"'.format(export_dir))
    lines.append('=' * 70)

    text = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print('\n已写入:', out_path)


if __name__ == '__main__':
    main()
