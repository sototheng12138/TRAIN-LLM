#!/usr/bin/env python3
"""
多样本/多通道一图多子图，输出优美 SVG。
支持：多个 export_dir（多样本）、多词集/多原型组（多通道），在单张图中以子图网格展示。

用法示例：
  # 单样本、3 个词集 → 1×3 子图
  python scripts/plot_multi_panel_heatmaps_svg.py --export_dir checkpoints/.../prototype_export --out multi_panel.svg

  # 多组原型（多通道）→ 2×2 或 3×2 子图
  python scripts/plot_multi_panel_heatmaps_svg.py --export_dir checkpoints/.../prototype_export --layout 2x2 --groups 0 1 2 3 --out multi_panel.svg

  # 多样本：多个 export_dir，每个样本一列
  python scripts/plot_multi_panel_heatmaps_svg.py --export_dirs dir1 dir2 dir3 --layout 1x3 --out multi_sample.svg
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 与 plot_prototype_word_heatmaps_business 一致的业务词集（用于横轴）
BUSINESS_WORD_SETS = [
    ('Time & horizon', [(11015, 'minute'), (17848, 'periodo'), (14448, 'Step'), (3706, 'times'), (18512, 'Sometimes'), (20202, 'termine')]),
    ('Volume & statistics', [(7977, 'volume'), (6124, 'addition'), (16538, 'sales'), (1788, 'system'), (16202, 'stats'), (16034, 'Status'), (9210, 'offset')]),
    ('Domain & business', [(7903, 'mine'), (13880, 'produces'), (16200, 'credit'), (1196, 'line'), (14407, 'underlying'), (7602, 'conv')]),
]


def get_parser():
    p = argparse.ArgumentParser(description='Multi-panel heatmaps (multi-sample / multi-channel) → SVG')
    p.add_argument('--export_dir', type=str, default='', help='单样本：prototype_export 目录')
    p.add_argument('--export_dirs', type=str, nargs='+', default=[], help='多样本：多个 prototype_export 目录')
    p.add_argument('--layout', type=str, default='1x3', help='子图网格 rows x cols，如 2x2, 3x2')
    p.add_argument('--groups', type=int, nargs='+', default=[], help='多通道：原型组号 0,1,2,... 每组 10 个原型；不指定则用单组 0')
    p.add_argument('--num_prototypes', type=int, default=10, help='每组原型数量')
    p.add_argument('--word_set_indices', type=int, nargs='+', default=[], help='只画哪几个词集 0,1,2；默认全部')
    p.add_argument('--out', type=str, default='', help='输出 SVG 路径；默认 export_dir/multi_panel_heatmaps.svg')
    p.add_argument('--dpi', type=int, default=150, help='若同时输出 PNG 时的 DPI')
    p.add_argument('--png', action='store_true', help='同时输出 PNG')
    p.add_argument('--no_cbar', action='store_true', help='不画共享 colorbar')
    p.add_argument('--labels', type=str, nargs='+', default=[], help='子图标签，如 a b c d；默认自动 (a)(b)...')
    p.add_argument('--font', type=str, default='', help='字体，如 DejaVu Sans, Times New Roman')
    return p


def _resolve_export_dirs(args):
    if args.export_dirs:
        return [d.rstrip('/') for d in args.export_dirs]
    if args.export_dir:
        return [args.export_dir.rstrip('/')]
    return []


def _find_matrix_path(export_dir):
    p = os.path.join(export_dir, 'prototype_word_weight_matrix.npy')
    if os.path.isfile(p):
        return p
    alt = os.path.join(export_dir, 'prototype_export', 'prototype_word_weight_matrix.npy')
    if os.path.isfile(alt):
        return alt
    return None


def load_W(export_dir):
    export_dir = export_dir.rstrip('/')
    path = _find_matrix_path(export_dir)
    if not path:
        raise FileNotFoundError('未找到 weight 矩阵: {}'.format(export_dir))
    return np.load(path)


def draw_one_heatmap(ax, W, word_list, num_prototypes, title, prototype_start=0, cmap='RdYlBu_r', vmin=None, vmax=None, use_shared_cbar=False):
    vocab_ids = [v for v, _ in word_list]
    labels = [lbl for _, lbl in word_list]
    end = min(prototype_start + num_prototypes, W.shape[0])
    protos = np.arange(prototype_start, end)
    if len(protos) == 0:
        return None
    W_sub = W[protos][:, np.array(vocab_ids)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    kwargs = dict(
        xticklabels=labels,
        yticklabels=[str(i) for i in protos],
        ax=ax,
        cmap=cmap,
        annot=False,
        cbar=not use_shared_cbar,
        cbar_kws={'label': 'weight', 'shrink': 0.6} if not use_shared_cbar else {},
    )
    if vmin is not None and vmax is not None:
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax
    else:
        kwargs['center'] = 0
    sns.heatmap(W_sub, **kwargs)
    ax.set_xlabel('')
    ax.set_ylabel('Prototype')
    ax.set_title(title, fontsize=11, pad=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    return W_sub


def _global_scale(W_list):
    if not W_list:
        return -1, 1
    flat = np.concatenate([W.ravel() for W in W_list])
    m, M = np.nanmin(flat), np.nanmax(flat)
    margin = max(0.1, (M - m) * 0.05)
    return m - margin, M + margin


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs
    import seaborn as sns

    export_dirs = _resolve_export_dirs(args)
    if not export_dirs:
        raise SystemExit('请指定 --export_dir 或 --export_dirs')

    # 解析 layout
    parts = args.layout.lower().split('x')
    if len(parts) != 2:
        raise SystemExit('--layout 应为 rowsxcols，如 1x3 或 2x2')
    n_rows, n_cols = int(parts[0]), int(parts[1])
    n_cells = n_rows * n_cols

    # 确定每个子图 (sample_idx, group, word_set_idx)
    groups = args.groups if args.groups else [0]
    word_set_indices = args.word_set_indices if args.word_set_indices else list(range(len(BUSINESS_WORD_SETS)))
    panels = []
    for sample_idx, exp_dir in enumerate(export_dirs):
        for g in groups:
            for ws_idx in word_set_indices:
                if len(panels) >= n_cells:
                    break
                panels.append((sample_idx, exp_dir, g, ws_idx))
            if len(panels) >= n_cells:
                break
        if len(panels) >= n_cells:
            break
    if not panels:
        raise SystemExit('没有可画的子图，请检查 --groups / --word_set_indices / --export_dirs')

    # 加载所有用到的 W，并收集所有 W_sub 用于统一 colorbar
    W_cache = {}
    all_W_sub = []
    for sample_idx, exp_dir, g, ws_idx in panels:
        if exp_dir not in W_cache:
            W_cache[exp_dir] = load_W(exp_dir)
        W = W_cache[exp_dir]
        _, word_list = BUSINESS_WORD_SETS[ws_idx]
        vocab_ids = [v for v, _ in word_list]
        proto_start = g * args.num_prototypes
        end = min(proto_start + args.num_prototypes, W.shape[0])
        if end <= proto_start:
            continue
        protos = np.arange(proto_start, end)
        W_sub = W[protos][:, np.array(vocab_ids)]
        all_W_sub.append(W_sub)
    vmin, vmax = _global_scale(all_W_sub)

    # 绘图风格：优美
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')
    if args.font:
        plt.rcParams['font.family'] = args.font
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3

    fig = plt.figure(figsize=(4.2 * n_cols, 3.8 * n_rows))
    grid = gs.GridSpec(n_rows, n_cols, figure=fig, hspace=0.32, wspace=0.28,
                       left=0.06, right=0.92 if not args.no_cbar else 0.98, top=0.92, bottom=0.10)

    use_shared_cbar = not args.no_cbar and len(all_W_sub) > 0
    if use_shared_cbar:
        cbar_ax = fig.add_axes([0.94, 0.12, 0.018, 0.76])
    else:
        cbar_ax = None

    label_iter = iter(args.labels) if args.labels else iter('abcdefghijklmnopqrstuvwxyz')
    for idx, (sample_idx, exp_dir, g, ws_idx) in enumerate(panels):
        row, col = idx // n_cols, idx % n_cols
        ax = fig.add_subplot(grid[row, col])
        W = W_cache[exp_dir]
        title, word_list = BUSINESS_WORD_SETS[ws_idx]
        sample_label = ''
        if len(export_dirs) > 1:
            sample_label = 'Sample {} '.format(sample_idx + 1)
        if len(groups) > 1:
            sample_label += 'P{}–{} '.format(g * args.num_prototypes, (g + 1) * args.num_prototypes - 1)
        sub_title = (sample_label + title).strip() or title
        draw_one_heatmap(ax, W, word_list, args.num_prototypes, sub_title,
                         prototype_start=g * args.num_prototypes,
                         vmin=vmin, vmax=vmax, use_shared_cbar=use_shared_cbar)
        try:
            letter = next(label_iter)
            ax.text(-0.06, 1.04, '({})'.format(letter), transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='bottom', ha='right')
        except StopIteration:
            pass

    if cbar_ax is not None:
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import matplotlib as mpl
        norm = Normalize(vmin=vmin, vmax=vmax)
        try:
            cmap = mpl.colormaps.get_cmap('RdYlBu_r')
        except AttributeError:
            cmap = plt.get_cmap('RdYlBu_r')
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(mappable, cax=cbar_ax)
        cb.set_label('weight', fontsize=10)
        cb.ax.tick_params(labelsize=9)

    out_path = args.out
    if not out_path:
        base = export_dirs[0]
        out_path = os.path.join(base, 'multi_panel_heatmaps.svg')
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print('已保存:', out_path)
    if args.png:
        png_path = os.path.splitext(out_path)[0] + '.png'
        fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight')
        print('已保存:', png_path)
    plt.close()


def main():
    args = get_parser().parse_args()
    run(args)


if __name__ == '__main__':
    main()
