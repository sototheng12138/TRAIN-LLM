#!/usr/bin/env python3
"""
贴合铁路货运/铁矿预测业务：用固定业务词集绘制「文本原型 × 词集」热力图。
输出多张图：时间与预测步、货运量与数值、领域与业务，词集无碎片、无跨集重复。

用法：
  cd /home/hesong/Time-LLM
  python scripts/plot_prototype_word_heatmaps_business.py --export_dir checkpoints/.../prototype_export
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# 业务词集：vocab_id -> 显示名（用于横轴）。仅包含与任务强相关的 token。
# 来自 prototype_topk_words.csv 中与「时间/预测步」「货运量/数值」「领域/业务」直接相关的词。
BUSINESS_WORD_SETS = [
    (
        'Time & horizon',
        [
            (11015, 'minute'),
            (17848, 'periodo'),
            (14448, 'Step'),
            (3706, 'times'),
            (18512, 'Sometimes'),
            (20202, 'termine'),
        ],
    ),
    (
        'Volume & statistics',
        [
            (7977, 'volume'),
            (6124, 'addition'),
            (16538, 'sales'),
            (1788, 'system'),
            (16202, 'stats'),
            (16034, 'Status'),
            (9210, 'offset'),
        ],
    ),
    (
        'Domain & business',
        [
            (7903, 'mine'),
            (13880, 'produces'),
            (16200, 'credit'),
            (1196, 'line'),
            (14407, 'underlying'),
            (7602, 'conv'),
        ],
    ),
]


def get_parser():
    p = argparse.ArgumentParser(description='Plot business-oriented prototype × word set heatmaps')
    p.add_argument('--export_dir', type=str, required=True, help='prototype_export 目录')
    p.add_argument('--num_prototypes', type=int, default=10, help='每组纵轴原型数量（每张图展示的原型个数）')
    p.add_argument('--group_start', type=int, default=0, help='组序号：0=原型0-9，1=10-19，…，9=90-99；与 num_prototypes 配合使用')
    p.add_argument('--all_groups', action='store_true', help='一次输出 10 组图（原型 0-9, 10-19, …, 90-99），文件名带 _g0 … _g9')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--separate', action='store_true', help='额外输出每词集单独一张图（默认只输出一张 3 子图）')
    p.add_argument('--cn', action='store_true', help='额外输出中文标题版（时间与预测步 / 货运量与数值 / 领域与业务）')
    p.add_argument('--font', type=str, default='', help='图内字体，如 Times New Roman 与正文统一；不指定则用 matplotlib 默认')
    p.add_argument('--out_dir', type=str, default='', help='图片输出目录，默认与 export_dir 相同；可指定如 final_prototype 将图写到该文件夹')
    p.add_argument('--title_suffix', type=str, default='', help='在图上方追加一行说明，如数据时间或 checkpoint 标识，便于区分不同 run')
    p.add_argument('--show_data_time', action='store_true', help='自动用 prototype_word_weight_matrix.npy 的修改时间作为 title_suffix，便于确认图来自哪次导出')
    p.add_argument('--show_data_hash', action='store_true', help='在图中追加 W 矩阵的数据指纹(SHA256 前12位)，权重不同则指纹不同，可确凿区分不同 checkpoint')
    return p


def load_data(export_dir):
    export_dir = export_dir.rstrip('/')
    W_path = os.path.join(export_dir, 'prototype_word_weight_matrix.npy')
    if not os.path.isfile(W_path):
        # 若传入的是 checkpoint 目录（上一级），尝试 prototype_export 子目录
        alt = os.path.join(export_dir, 'prototype_export', 'prototype_word_weight_matrix.npy')
        if os.path.isfile(alt):
            export_dir = os.path.join(export_dir, 'prototype_export')
            W_path = alt
        else:
            raise FileNotFoundError(
                '未找到 prototype_word_weight_matrix.npy。请把 --export_dir 设为「含该文件的目录」的完整路径，例如：\n'
                '  --export_dir checkpoints/long_term_forecast_Iron_96_48_TimeLLM_..._0-iron/prototype_export'
            )
    return np.load(W_path), export_dir


def draw_one_heatmap(ax, W, word_list, num_prototypes, title, prototype_start=0):
    """word_list: [(vocab_id, label), ...]. prototype_start: 纵轴起始原型索引（用于多组）。"""
    vocab_ids = [v for v, _ in word_list]
    labels = [lbl for _, lbl in word_list]
    end = min(prototype_start + num_prototypes, W.shape[0])
    protos = np.arange(prototype_start, end)
    if len(protos) == 0:
        return False
    W_sub = W[protos][:, np.array(vocab_ids)]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return False
    sns.heatmap(
        W_sub,
        xticklabels=labels,
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
    return True


def _run_one_group(W, out_dir, args, group_start, suffix):
    """为某一组原型（group_start*10 起共 num_prototypes 个）出图。suffix 如 '' 或 '_g1'。"""
    n_proto = args.num_prototypes
    proto_start = group_start * n_proto
    if proto_start >= W.shape[0]:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    titles_en = [t for t, _ in BUSINESS_WORD_SETS]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (_, word_list), title in zip(axes, BUSINESS_WORD_SETS, titles_en):
        draw_one_heatmap(ax, W, word_list, n_proto, title, prototype_start=proto_start)
    for ax in axes:
        ax.set_title('')
    if getattr(args, 'title_suffix', ''):
        fig.suptitle(args.title_suffix, fontsize=9, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'prototype_word_sets_heatmaps_business{}.png'.format(suffix))
    out_svg = os.path.join(out_dir, 'prototype_word_sets_heatmaps_business{}.svg'.format(suffix))
    plt.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    print('已保存:', out_png, ',', out_svg)
    if args.separate:
        for i, (title, word_list) in enumerate(BUSINESS_WORD_SETS):
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            draw_one_heatmap(ax, W, word_list, n_proto, title, prototype_start=proto_start)
            plt.tight_layout()
            out_path = os.path.join(out_dir, 'prototype_word_set_business_{}{}.png'.format(i + 1, suffix))
            plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
            plt.close()
            print('已保存:', out_path)
    if args.cn:
        titles_cn = ['时间与预测步', '货运量与数值', '领域与业务']
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (_, word_list), title in zip(axes, BUSINESS_WORD_SETS, titles_cn):
            draw_one_heatmap(ax, W, word_list, n_proto, title, prototype_start=proto_start)
        for ax in axes:
            ax.set_title('')
        if getattr(args, 'title_suffix', ''):
            fig.suptitle(args.title_suffix, fontsize=9, y=1.02)
        plt.tight_layout()
        out_cn_png = os.path.join(out_dir, 'prototype_word_sets_heatmaps_business_cn{}.png'.format(suffix))
        out_cn_svg = os.path.join(out_dir, 'prototype_word_sets_heatmaps_business_cn{}.svg'.format(suffix))
        plt.savefig(out_cn_png, dpi=args.dpi, bbox_inches='tight')
        plt.savefig(out_cn_svg, format='svg', bbox_inches='tight')
        plt.close()
        print('已保存:', out_cn_png, ',', out_cn_svg)


def main():
    args = get_parser().parse_args()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print('请安装 matplotlib 与 seaborn: pip install matplotlib seaborn')
        return

    if args.font:
        plt.rcParams['font.family'] = args.font
        plt.rcParams['axes.unicode_minus'] = False
    W, export_dir = load_data(args.export_dir)
    if getattr(args, 'show_data_time', False):
        npy_path = os.path.join(export_dir, 'prototype_word_weight_matrix.npy')
        if os.path.isfile(npy_path):
            from datetime import datetime
            mtime = datetime.fromtimestamp(os.path.getmtime(npy_path))
            args.title_suffix = (args.title_suffix or '') + ('  [W from %s]' % mtime.strftime('%Y-%m-%d %H:%M'))
    if getattr(args, 'show_data_hash', False):
        import hashlib
        h = hashlib.sha256(W[:20, :100].tobytes()).hexdigest()[:12]
        args.title_suffix = (args.title_suffix or '') + ('  [hash %s]' % h)
    out_dir = (args.out_dir or export_dir).rstrip('/')
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print('已创建输出目录:', out_dir)
    n_proto = args.num_prototypes
    n_total = W.shape[0]

    if args.all_groups:
        # 10 组：原型 0-9, 10-19, …, 90-99
        for g in range(10):
            if g * n_proto >= n_total:
                break
            suffix = '_g{}'.format(g)
            print('组 {}: 原型 {} - {}'.format(g, g * n_proto, min(g * n_proto + n_proto, n_total) - 1))
            _run_one_group(W, out_dir, args, g, suffix)
        return

    # 单组：默认 group_start=0（原型 0-9），与原先行为一致
    suffix = '_g{}'.format(args.group_start) if args.group_start else ''
    _run_one_group(W, out_dir, args, args.group_start, suffix)


if __name__ == '__main__':
    main()
