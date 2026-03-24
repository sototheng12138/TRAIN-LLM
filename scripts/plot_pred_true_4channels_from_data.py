#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Eval 保存的 pred_true_4channels_data.npz 直接重绘四通道预测图，无需重跑评估。
用法:
  python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir /path/to/checkpoint_dir
  python scripts/plot_pred_true_4channels_from_data.py --data_file /path/to/pred_true_4channels_data.npz --out_dir /path/to/output
  # 论文风格（推荐）
  python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir /path/to/checkpoint_dir --style paper --out_name pred_true_4channels_paper
改图时只需编辑本脚本中的绘图逻辑，然后运行上述命令即可。
"""
import argparse
import os
import numpy as np


def _apply_paper_style(plt):
    # Prefer SciencePlots if available (IEEE/Nature-like). Fallback is a custom
    # publication-friendly rcParams set.
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'ieee'])
    except Exception:
        plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # Font: try Times New Roman first, fallback to common serif.
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        # Sizing: ensure readability in single-column figures (min >= 8pt).
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        # Spines are configured later (open/closed) per user preference.
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.grid': True,
        'grid.color': '#cfcfcf',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.30,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.9,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.transparent': False,
        'svg.fonttype': 'none',  # keep text as text in SVG
    })


def _sanitize_title(s: str) -> str:
    return str(s).strip()

def _scale_for_label(y: np.ndarray):
    """
    Avoid tiny offset text like '1e6' by scaling data and writing the magnitude
    directly into the axis label.

    Returns: (y_scaled, label_suffix)
      - label_suffix: '', '(×10^k)', or '(Millions)'
    """
    y = np.asarray(y, dtype=float)
    max_abs = float(np.nanmax(np.abs(y))) if y.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        return y, ''

    # Choose exponent in steps of 3 for readability.
    exp = int(np.floor(np.log10(max_abs) / 3) * 3)
    exp = max(0, min(exp, 12))
    if exp == 0:
        return y, ''
    y_scaled = y / (10 ** exp)
    if exp == 6:
        return y_scaled, '(Millions)'
    if exp == 9:
        return y_scaled, '(Billions)'
    return y_scaled, rf'($\times 10^{{{exp}}}$)'

def _choose_global_scale(true_first: np.ndarray, pred_first: np.ndarray, n_plot: int):
    """
    Choose one shared magnitude for all subplots so y-units are consistent.
    Returns (scale_factor, y_label).
    """
    y = np.concatenate([true_first[:, :n_plot].reshape(-1), pred_first[:, :n_plot].reshape(-1)], axis=0)
    y = np.asarray(y, dtype=float)
    max_abs = float(np.nanmax(np.abs(y))) if y.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        return 1.0, 'Shipment'

    exp = int(np.floor(np.log10(max_abs) / 3) * 3)
    exp = max(0, min(exp, 12))
    if exp == 0:
        return 1.0, 'Shipment'
    if exp == 6:
        return 1e6, 'Shipment (Millions)'
    if exp == 9:
        return 1e9, 'Shipment (Billions)'
    return float(10 ** exp), rf'Shipment ($\times 10^{{{exp}}}$)'


def main():
    parser = argparse.ArgumentParser(description='从 npz 重绘 pred_true_4channels 图')
    parser.add_argument('--ckpt_dir', type=str, default='', help='checkpoint 目录，其下应有 pred_true_4channels_data.npz')
    parser.add_argument('--data_file', type=str, default='', help='直接指定 .npz 路径（与 --ckpt_dir 二选一）')
    parser.add_argument('--out_dir', type=str, default='', help='输出目录，默认与数据文件同目录')
    parser.add_argument('--dpi', type=int, default=180, help='PNG 分辨率')
    parser.add_argument('--style', type=str, default='paper', choices=['paper', 'default'], help='出图风格: paper=期刊风格; default=基础风格')
    parser.add_argument('--out_name', type=str, default='pred_true_4channels', help='输出文件名（不含扩展名）')
    parser.add_argument('--n_plot', type=int, default=4, help='绘制前 N 个通道（默认 4；如 10 通道任务可设 10）')
    parser.add_argument('--ncols', type=int, default=0, help='子图列数（0=自动：<=4 用 2 列，否则用 5 列）')
    parser.add_argument('--markers', action='store_true', help='在 True 上添加小标记（黑白打印更易区分）')
    parser.add_argument('--markevery', type=int, default=5, help='marker 间隔（例如 5 表示每 5 个点一个 marker）')
    parser.add_argument(
        '--palette',
        type=str,
        default='print',
        choices=['print', 'color', 'classic'],
        help='配色: print=黑白/灰度友好; color=色盲友好; classic=经典学术蓝/橙(Tab10)'
    )
    parser.add_argument('--spines', type=str, default='open', choices=['open', 'closed'], help='边框: open=去掉上/右; closed=四面闭合')
    args = parser.parse_args()

    if args.data_file:
        data_path = args.data_file
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(data_path))
    elif args.ckpt_dir:
        data_path = os.path.join(args.ckpt_dir, 'pred_true_4channels_data.npz')
        out_dir = args.out_dir or args.ckpt_dir
    else:
        print('请指定 --ckpt_dir 或 --data_file')
        return

    if not os.path.isfile(data_path):
        print('未找到数据文件:', data_path)
        return

    data = np.load(data_path, allow_pickle=True)
    pred_first = data['pred_first']   # (pred_len, n_channels)
    true_first = data['true_first']   # (pred_len, n_channels)
    pred_len = int(data['pred_len'])
    cn = data['col_names']
    col_names = cn.tolist() if cn.ndim >= 1 and cn.size > 1 else [str(cn.flat[0])]

    n_plot = int(max(1, min(getattr(args, 'n_plot', 4), pred_first.shape[1])))
    if len(col_names) > n_plot:
        col_names = col_names[:n_plot]
    while len(col_names) < n_plot:
        col_names.append(f'Ch{len(col_names)}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if args.style == 'paper':
        _apply_paper_style(plt)

    # Figure layout
    ncols = int(getattr(args, 'ncols', 0) or 0)
    if ncols <= 0:
        ncols = 2 if n_plot <= 4 else 5
    nrows = int(np.ceil(n_plot / ncols))
    # heuristic sizing: keep similar density across 2x2 and 2x5 figures
    if n_plot <= 4:
        figsize = (7.2, 6.0)  # single-column friendly
    else:
        figsize = (min(13.5, 2.4 * ncols), min(9.5, 3.0 * nrows + 1.0))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=250,
        sharex=True,
    )
    axes = np.array(axes).reshape(-1)
    t = np.arange(pred_len)
    # Palette
    if args.palette == 'print':
        # High contrast in grayscale printing
        color_true, color_pred = '#111111', '#666666'
    elif args.palette == 'classic':
        # Classic academic look (Matplotlib Tab10 default)
        # In grayscale, orange tends to be lighter than blue, which helps separation.
        color_true, color_pred = '#1f77b4', '#ff7f0e'
    else:
        # Colorblind-friendly (Okabe-Ito)
        color_true, color_pred = '#0072B2', '#D55E00'

    # 通道名中文 -> 英文（若 col_names 已是英文则保留）
    name_en = {'铁矿石': 'Iron ore', '铁矿砂': 'Iron sand', '铁矿粉': 'Iron powder', '铁精矿粉': 'Iron concentrate'}
    titles = [name_en.get(str(n), str(n)) for n in (col_names[:n_plot] if len(col_names) >= n_plot else col_names)]
    titles = [_sanitize_title(t) for t in titles]
    while len(titles) < n_plot:
        titles.append('Ch{}'.format(len(titles)))

    # Plot styling
    lw_true, lw_pred = 2.1, 2.3  # make Pred slightly thicker for print
    marker = 'o' if args.markers else None
    markevery = max(1, int(args.markevery))

    # We share x: show x labels only on bottom row
    # Use a single y-unit across all subplots (journal-friendly).
    scale_factor, y_label = _choose_global_scale(true_first, pred_first, n_plot=n_plot)

    for c in range(n_plot):
        ax = axes[c]
        y_true = true_first[:, c] / scale_factor
        y_pred = pred_first[:, c] / scale_factor

        ax.plot(
            t, y_true,
            color=color_true, linestyle='-',
            label='True', alpha=0.95,
            linewidth=lw_true,
            marker=marker, markersize=3.2,
            markerfacecolor=color_true,
            markeredgewidth=0.0,
            markevery=markevery if marker else None,
        )
        ax.plot(
            t, y_pred,
            color=color_pred, linestyle='--',
            label='Pred', alpha=0.90,
            linewidth=lw_pred,
            # No markers on Pred by default (keeps plot clean; True markers suffice)
        )
        ax.set_title(titles[c], fontsize=12, fontweight='medium')
        # show x labels only on bottom row
        if c < (nrows - 1) * ncols:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Forecast step')

        # Remove per-axes y label, use a single shared y label on the figure.
        ax.set_ylabel('')

        # Light, unobtrusive grid: only horizontal lines
        ax.grid(True, axis='y', which='major')
        ax.grid(False, axis='x')
        ax.tick_params(axis='both')
        ax.set_xlim(-0.5, pred_len - 0.5)
        # Avoid matplotlib offset text (1e6). We scale data + label magnitude instead.
        ax.get_yaxis().get_offset_text().set_visible(False)
        # Spines style
        if args.spines == 'open':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)

    # Hide unused axes (when n_plot doesn't fill the grid)
    for k in range(n_plot, nrows * ncols):
        axes[k].axis('off')

    # Shared y label (saves space) with explicit unit.
    fig.supylabel(y_label, x=0.02)

    # Single legend for the whole figure (top-center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.2,
        columnspacing=1.5,
    )

    # Tight layout with controlled spacing
    if n_plot <= 4:
        fig.subplots_adjust(top=0.88, left=0.10, right=0.99, bottom=0.08, hspace=0.18, wspace=0.25)
    else:
        fig.subplots_adjust(top=0.90, left=0.07, right=0.995, bottom=0.08, hspace=0.25, wspace=0.25)

    plot_png = os.path.join(out_dir, f'{args.out_name}.png')
    plot_svg = os.path.join(out_dir, f'{args.out_name}.svg')
    plot_pdf = os.path.join(out_dir, f'{args.out_name}.pdf')
    fig.savefig(plot_png, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_svg, format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig(plot_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print('已重绘:', plot_png, ',', plot_svg, ',', plot_pdf)


if __name__ == '__main__':
    main()
