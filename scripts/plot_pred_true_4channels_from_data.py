#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Eval 保存的 pred_true_4channels_data.npz 直接重绘四通道预测图，无需重跑评估。
用法:
  python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir /path/to/checkpoint_dir
  python scripts/plot_pred_true_4channels_from_data.py --data_file /path/to/pred_true_4channels_data.npz --out_dir /path/to/output
改图时只需编辑本脚本中的绘图逻辑，然后运行上述命令即可。
"""
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='从 npz 重绘 pred_true_4channels 图')
    parser.add_argument('--ckpt_dir', type=str, default='', help='checkpoint 目录，其下应有 pred_true_4channels_data.npz')
    parser.add_argument('--data_file', type=str, default='', help='直接指定 .npz 路径（与 --ckpt_dir 二选一）')
    parser.add_argument('--out_dir', type=str, default='', help='输出目录，默认与数据文件同目录')
    parser.add_argument('--dpi', type=int, default=180, help='PNG 分辨率')
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

    n_plot = min(4, pred_first.shape[1])
    if len(col_names) > n_plot:
        col_names = col_names[:n_plot]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 图中全部使用英文，避免字体/乱码问题
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
    axes = axes.flatten()
    t = np.arange(pred_len)
    color_true, color_pred = '#1f77b4', '#d62728'

    # 通道名中文 -> 英文（若 col_names 已是英文则保留）
    name_en = {'铁矿石': 'Iron ore', '铁矿砂': 'Iron sand', '铁矿粉': 'Iron powder', '铁精矿粉': 'Iron concentrate'}
    titles = [name_en.get(str(n), str(n)) for n in (col_names[:n_plot] if len(col_names) >= n_plot else col_names)]
    while len(titles) < n_plot:
        titles.append('Ch{}'.format(len(titles)))

    for c in range(n_plot):
        ax = axes[c]
        ax.plot(t, true_first[:, c], color=color_true, linestyle='-', label='True', alpha=0.95, linewidth=2.2)
        ax.plot(t, pred_first[:, c], color=color_pred, linestyle='--', label='Pred', alpha=0.9, linewidth=1.5)
        ax.set_title(titles[c], fontsize=12, fontweight='medium')
        ax.set_xlabel('Forecast step', fontsize=10)
        ax.set_ylabel('Shipment (original scale)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, which='both', alpha=0.25, linestyle='-')
        ax.tick_params(axis='both', labelsize=9)
        ax.set_xlim(-0.5, pred_len - 0.5)

    plt.tight_layout()

    plot_png = os.path.join(out_dir, 'pred_true_4channels.png')
    plot_svg = os.path.join(out_dir, 'pred_true_4channels.svg')
    fig.savefig(plot_png, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_svg, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print('已重绘:', plot_png, ',', plot_svg)


if __name__ == '__main__':
    main()
