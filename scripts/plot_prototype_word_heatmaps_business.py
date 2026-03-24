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


# 业务词集：token id(s) -> 显示名（用于横轴）。
# - 单词在 LLaMA tokenizer 下可能被拆成多个 sub-tokens（例如 freight, shortage），此时用多个 vocab_id 聚合成一列。
# - 仅包含与任务强相关的 token（时间周期 / 零膨胀与峰值 / 铁路与矿业实体）。
BUSINESS_WORD_SETS = [
    (
        'Temporal & periodic dynamics',
        [
            (931, 'time'),
            (3785, 'period'),
            (14218, 'daily'),
            (11015, 'minute'),
            (20410, 'schedule'),
            (4259, 'season'),
            (9055, 'delay'),
        ],
    ),
    (
        'Volume & extreme variations',
        [
            (7977, 'volume'),
            (5225, 'zero'),
            (4069, 'empty'),
            (19224, 'peak'),
            (2254, 'load'),
            (13284, 'capacity'),
            ([3273, 482], 'shortage'),
        ],
    ),
    (
        'Railway & mining entities',
        [
            (7903, 'mine'),
            (7945, 'train'),
            ([3005, 523], 'freight'),
            (5073, 'station'),
            (5782, 'route'),
            (13916, 'dispatch'),
            (17040, 'cargo'),
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
    """word_list: [(vocab_id_or_list, label), ...]. prototype_start: 纵轴起始原型索引（用于多组）。

    vocab_id_or_list:
      - int: single token column in W
      - list[int]: multiple tokens, aggregated to one column by mean()
    """
    # Build an aggregated sub-matrix: (num_prototypes, n_words)
    labels = [lbl for _, lbl in word_list]
    end = min(prototype_start + num_prototypes, W.shape[0])
    protos = np.arange(prototype_start, end)
    if len(protos) == 0:
        return False
    cols = []
    for v, _ in word_list:
        if isinstance(v, (list, tuple, np.ndarray)):
            ids = [int(x) for x in v]
            cols.append(np.mean(W[protos][:, np.array(ids)], axis=1, keepdims=True))
        else:
            cols.append(W[protos][:, np.array([int(v)])])
    W_sub = np.concatenate(cols, axis=1) if cols else W[protos][:, :0]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return False
    # NOTE: Keep this helper minimal; styling is handled by the multi-panel plot below.
    sns.heatmap(
        W_sub,
        xticklabels=labels,
        yticklabels=[str(i) for i in protos],
        ax=ax,
        cmap='RdBu_r',
        center=0,
        cbar=False,
        annot=False,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    return True


def _run_one_group(W, out_dir, args, group_start, suffix):
    """为某一组原型（group_start*10 起共 num_prototypes 个）出图。suffix 如 '' 或 '_g1'。"""
    n_proto = args.num_prototypes
    proto_start = group_start * n_proto
    if proto_start >= W.shape[0]:
        return
    end = min(proto_start + n_proto, W.shape[0])
    protos = np.arange(proto_start, end)
    if protos.size == 0:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    # ---------- Publication-style multi-panel heatmap (shared colorbar, clean layout) ----------
    # Global style (serif, minimal clutter)
    # Style first (may override fonts), then enforce Times-like serif fonts.
    try:
        plt.style.use('seaborn-v0_8-white')
    except Exception:
        pass
    # Font: prefer Times-like fonts actually present on most Linux installs.
    # On this machine, Nimbus Roman + Liberation Serif are available and are Times-compatible.
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times Roman', 'Nimbus Roman', 'Liberation Serif', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False,
    })

    # Build three sub-matrices with shared scale
    W_rows = W[protos]  # (num_prototypes, vocab)
    mats = []
    xlabels = []
    for _, word_list in BUSINESS_WORD_SETS:
        labels = [lbl for _, lbl in word_list]
        cols = []
        for v, _ in word_list:
            if isinstance(v, (list, tuple, np.ndarray)):
                ids = [int(x) for x in v]
                cols.append(np.mean(W_rows[:, np.array(ids)], axis=1, keepdims=True))
            else:
                cols.append(W_rows[:, np.array([int(v)])])
        W_sub = np.concatenate(cols, axis=1) if cols else W_rows[:, :0]
        mats.append(W_sub)
        xlabels.append(labels)

    # Scientific-notation optimization: scale weights by 1e3 for display
    mats_scaled = [m * 1e3 for m in mats]
    max_abs = float(np.nanmax(np.abs(np.concatenate([m.ravel() for m in mats_scaled], axis=0)))) if mats_scaled else 1.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    vmax = max_abs
    vmin = -max_abs

    # Layout: three heatmaps tightly + one shared colorbar on the right
    fig = plt.figure(figsize=(10.6, 3.2), dpi=250)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.06, left=0.06, right=0.98, top=0.92, bottom=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    # Subtitles: remove code traces, use (a)(b)(c)
    subtitles_en = ['(a) Time features', '(b) Capacity features', '(c) Logistics features']
    subtitles_cn = ['(a) 时间特征', '(b) 运力特征', '(c) 物流特征']

    cmap = 'RdBu_r'  # cleaner white-ish center than RdYlBu_r
    ytick = [str(i) for i in protos]
    for i, ax in enumerate(axes):
        show_y = (i == 0)
        hm = sns.heatmap(
            mats_scaled[i],
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0.0,
            xticklabels=xlabels[i],
            yticklabels=ytick if show_y else False,
            cbar=(i == 2),
            cbar_ax=cax if i == 2 else None,
            linewidths=0.5,
            linecolor='white',
            annot=False,
        )
        # X labels: rotated with right alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('')
        if show_y:
            ax.set_ylabel('Prototype')
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False)
        # Remove repeated y tick labels for non-left panels
        if not show_y:
            ax.set_yticks([])
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Panel subtitle (no global title)
        ax.set_title(subtitles_en[i], pad=6)

    # Shared colorbar formatting: simpler ticks + explicit scale
    cb = hm.collections[0].colorbar
    cb.set_label(r'Weight ($\times 10^{-3}$)', fontsize=11)
    # Prefer symmetric ticks like -4,-2,0,2,4 when range allows
    tick_max = max(2, int(np.ceil(max_abs / 2) * 2))
    ticks = np.linspace(-tick_max, tick_max, 5)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=10)

    out_png = os.path.join(out_dir, f'prototype_word_sets_heatmaps_business{suffix}.png')
    out_svg = os.path.join(out_dir, f'prototype_word_sets_heatmaps_business{suffix}.svg')
    out_pdf = os.path.join(out_dir, f'prototype_word_sets_heatmaps_business{suffix}.pdf')
    fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(out_svg, format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('已保存:', out_png, ',', out_svg, ',', out_pdf)
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
        # Chinese font fallback (avoid tofu squares). Use common CJK fonts if available.
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [
                'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC',
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC',
                'DejaVu Sans',
            ],
            'axes.unicode_minus': False,
        })
        # Chinese subtitle version (same layout; no global title)
        fig = plt.figure(figsize=(10.6, 3.2), dpi=250)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.06, left=0.06, right=0.98, top=0.92, bottom=0.22)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])

        hm = None
        for i, ax in enumerate(axes):
            show_y = (i == 0)
            hm = sns.heatmap(
                mats_scaled[i],
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=0.0,
                xticklabels=xlabels[i],
                yticklabels=ytick if show_y else False,
                cbar=(i == 2),
                cbar_ax=cax if i == 2 else None,
                linewidths=0.5,
                linecolor='white',
                annot=False,
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlabel('')
            if show_y:
                ax.set_ylabel('Prototype')
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', left=False)
                ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(subtitles_cn[i], pad=6)

        cb = hm.collections[0].colorbar
        cb.set_label(r'Weight ($\times 10^{-3}$)', fontsize=11)
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=10)

        out_cn_png = os.path.join(out_dir, f'prototype_word_sets_heatmaps_business_cn{suffix}.png')
        out_cn_svg = os.path.join(out_dir, f'prototype_word_sets_heatmaps_business_cn{suffix}.svg')
        out_cn_pdf = os.path.join(out_dir, f'prototype_word_sets_heatmaps_business_cn{suffix}.pdf')
        fig.savefig(out_cn_png, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(out_cn_svg, format='svg', bbox_inches='tight', facecolor='white')
        fig.savefig(out_cn_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('已保存:', out_cn_png, ',', out_cn_svg, ',', out_cn_pdf)


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
