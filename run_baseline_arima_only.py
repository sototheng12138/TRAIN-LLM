#!/usr/bin/env python3
"""
仅运行 ARIMA 基线，与 run_baseline_xgb_arima.py 使用相同的数据划分与评估方式。
结果可单独用于补充 baseline_eval_result.txt 中的 ARIMA 部分。

依赖: pip install pandas numpy scikit-learn statsmodels
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- 配置（与 run_baseline_xgb_arima 一致）----------
SEQ_LEN = 96
PRED_LEN = 48
ROOT = './dataset/'
DATA_PATH = '2023_2025_Iron_data.csv'
COLS = ['铁矿石', '铁矿砂', '铁矿粉', '铁精矿粉']


def load_data():
    path = os.path.join(ROOT, DATA_PATH)
    df = pd.read_csv(path)
    if '日期' in df.columns:
        df = df.rename(columns={'日期': 'date'})
    df = df.sort_values('date').reset_index(drop=True)
    return df


def run_arima_rolling(ts_train, ts_test, pre_len, train_std, max_windows=None):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return None, None, None
    n_test = len(ts_test)
    if n_test < pre_len:
        return None, None, None
    n_windows = min(n_test - pre_len + 1, max_windows or n_test)
    preds, trues = [], []
    for i in range(n_windows):
        history = np.concatenate([ts_train, ts_test[:i]]) if i > 0 else ts_train.copy()
        if len(history) < 30:
            continue
        try:
            model = ARIMA(history, order=(2, 1, 2))
            fit = model.fit()
            f = fit.forecast(steps=pre_len)
            preds.append(f)
            trues.append(ts_test[i : i + pre_len])
        except Exception:
            continue
    if len(preds) == 0:
        return None, None, None
    preds = np.array(preds)
    trues = np.array(trues)
    mae_orig = mean_absolute_error(trues, preds)
    rmse_orig = np.sqrt(mean_squared_error(trues, preds))
    mae_s = mae_orig / train_std if train_std > 0 else np.nan
    rmse_s = rmse_orig / train_std if train_std > 0 else np.nan
    return mae_s, rmse_s, len(preds)


def main():
    parser = argparse.ArgumentParser(description='仅运行 ARIMA 基线')
    parser.add_argument('--quick', action='store_true', help='仅用前 30 个测试窗口以省时')
    args = parser.parse_args()
    max_windows = None if not args.quick else 30

    df = load_data()
    total_len = len(df)
    num_train = int(total_len * 0.7)
    num_test = int(total_len * 0.2)
    test_start = total_len - num_test
    n_test_windows = num_test - PRED_LEN + 1

    print('=' * 60)
    print(' ARIMA 基线（与 run_baseline_xgb_arima 相同设置）')
    print('=' * 60)
    print(' 总长度={}, 训练={}, 测试={}, 测试窗口数={}'.format(total_len, num_train, num_test, n_test_windows))
    if max_windows:
        print(' [--quick] 仅用前 {} 个窗口'.format(max_windows))
    print()

    by_channel = []
    for col in COLS:
        ts = df[col].astype(float).values
        train_ts = ts[:num_train]
        test_ts = ts[test_start:]
        scaler = StandardScaler()
        scaler.fit(train_ts.reshape(-1, 1))
        train_std = np.sqrt(scaler.var_[0])
        mae_s, rmse_s, n_win = run_arima_rolling(
            train_ts, test_ts, PRED_LEN, train_std, max_windows=max_windows
        )
        if mae_s is None:
            print(' [{}] ARIMA 失败（statsmodels 未安装或全部窗口拟合失败）'.format(col))
            continue
        by_channel.append((mae_s, rmse_s))
        print(' [{}] 标准化 MAE = {:.6f}, RMSE = {:.6f} ({} 窗口)'.format(col, mae_s, rmse_s, n_win))

    if not by_channel:
        print()
        print(' ARIMA 无有效结果，请确认 pip install statsmodels')
        print('=' * 60)
        return

    mae_all = np.mean([p[0] for p in by_channel])
    rmse_all = np.sqrt(np.mean(np.array([p[1] for p in by_channel]) ** 2))

    print()
    print('【与 eval_result.txt 可比】')
    print('  ARIMA 整体 标准化 MAE = {:.6f}  RMSE = {:.6f}'.format(mae_all, rmse_all))
    print()
    print(' 主模型 iron 标准化: MAE ≈ 0.7747, RMSE ≈ 1.2371')
    print('=' * 60)

    # 写入 arima_eval_result.txt，便于手动补充到 baseline_eval_result.txt
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arima_eval_result.txt')
    lines = []
    lines.append('模型: ARIMA')
    lines.append('Test 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(mae_all, rmse_all))
    lines.append('各指标:')
    for i, col in enumerate(COLS):
        if i < len(by_channel):
            lines.append('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(col, by_channel[i][0], by_channel[i][1]))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(' ARIMA 结果已写入: {}'.format(out_path))
    print(' 可将上述内容替换 baseline_eval_result.txt 中的「模型: ARIMA  (未运行或依赖未安装)」段落')


if __name__ == '__main__':
    main()
