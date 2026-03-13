#!/usr/bin/env python3
"""
主模型评估：趋势方向混淆矩阵（预测期前半段 vs 后半段均值 → 上升/持平/下降）。
需先运行评估并保存 pred/true：
  python run_eval.py --save_pred_true
  或 bash scripts/Eval_Iron.sh 时在 run_eval.py 中加上 --save_pred_true
然后指定 checkpoint 目录运行本脚本，生成优雅的混淆矩阵图。
"""
import argparse
import os
import numpy as np

# 趋势离散化：预测期前半 vs 后半均值，相对变化超过 thresh_ratio 视为上升/下降
TREND_THRESH_RATIO = 0.1  # 10% 相对变化以内视为持平
LABELS = ['下降', '持平', '上升']  # 对应 -1, 0, 1


def get_trend(seq: np.ndarray, thresh_ratio: float = TREND_THRESH_RATIO) -> int:
    """seq: (T,) 或 (T,C) 取最后一维的均值。返回 -1/0/1。"""
    if seq.size == 0:
        return 0
    s = np.asarray(seq).reshape(-1)
    half = len(s) // 2
    m1, m2 = np.mean(s[:half]), np.mean(s[half:])
    scale = np.std(s) + 1e-8
    if scale < 1e-8:
        return 0
    change = (m2 - m1) / scale
    if change >= thresh_ratio:
        return 1
    if change <= -thresh_ratio:
        return -1
    return 0


def build_confusion_matrix(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred/true: (n_windows, pred_len, n_channels). 返回 3x3 (true 为行, pred 为列)。"""
    cm = np.zeros((3, 3), dtype=np.int64)
    n_w, _, n_c = pred.shape
    for w in range(n_w):
        for c in range(n_c):
            t_true = get_trend(true[w, :, c])
            t_pred = get_trend(pred[w, :, c])
            # 映射 -1,0,1 -> 0,1,2
            i = t_true + 1
            j = t_pred + 1
            cm[i, j] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description='Plot trend-direction confusion matrix from saved pred/true')
    parser.add_argument('--ckpt_dir', type=str, default='',
                        help='checkpoint 目录（含 pred.npy, true.npy）；默认用主模型 iron')
    parser.add_argument('--out', type=str, default='', help='输出图片路径，默认存到 ckpt_dir/confusion_matrix.png')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not args.ckpt_dir:
        # 默认主模型 iron
        args.ckpt_dir = os.path.join(root, 'checkpoints',
            'long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron')
    pred_path = os.path.join(args.ckpt_dir, 'pred.npy')
    true_path = os.path.join(args.ckpt_dir, 'true.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print('未找到 pred.npy 或 true.npy，请先运行: python run_eval.py --save_pred_true')
        print('或修改 scripts/Eval_Iron.sh 在 run_eval.py 后加 --save_pred_true 后重新评估。')
        print('期望目录:', args.ckpt_dir)
        return

    pred = np.load(pred_path)
    true = np.load(true_path)
    # 形状应为 (n_windows, pred_len, n_channels)
    if pred.ndim == 2:
        pred = pred[:, :, np.newaxis]
    if true.ndim == 2:
        true = true[:, :, np.newaxis]

    cm = build_confusion_matrix(pred, true)
    total = cm.sum()
    acc = np.diag(cm).sum() / total if total > 0 else 0

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print('请安装 matplotlib 与 seaborn: pip install matplotlib seaborn')
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=LABELS, yticklabels=LABELS,
                cbar_kws={'label': '样本数'}, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('预测趋势', fontsize=11)
    ax.set_ylabel('真实趋势', fontsize=11)
    ax.set_title('主模型 预测期趋势方向 混淆矩阵\n(准确率 {:.1%})'.format(acc), fontsize=12)
    plt.tight_layout()

    out_path = args.out or os.path.join(args.ckpt_dir, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print('已保存:', out_path)


if __name__ == '__main__':
    main()
