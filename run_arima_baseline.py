"""
ARIMA 基线：与 run_eval 相同的数据划分与评估方式，计算标准化空间和逆变换后的 MAE/RMSE。
用法：在 Time-LLM 目录下运行
  python run_arima_baseline.py           # 默认只跑前 20 个窗口，较快
  python run_arima_baseline.py --full     # 跑全部 172 个窗口，较慢
依赖：pip install statsmodels
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 与 data_loader 中一致
SEQ_LEN = 96
PRED_LEN = 48
ROOT_PATH = './dataset/'
DATA_PATH = '2023_2025_Iron_data.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='跑全部窗口（慢）；默认只跑前 20 个窗口')
    args_cli = parser.parse_args()
    path = os.path.join(ROOT_PATH, DATA_PATH)
    df_raw = pd.read_csv(path)
    if '日期' in df_raw.columns:
        df_raw = df_raw.rename(columns={'日期': 'date'})

    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    # 与 Dataset_Custom_Iron 一致：test 段从 (len - num_test - seq_len) 开始，保证第一个窗口有 seq_len 输入
    test_start = len(df_raw) - num_test - SEQ_LEN  # 781
    # 第一个要预测的起点：test 段内第 SEQ_LEN 个点 = 全局 test_start + SEQ_LEN
    forecast_start = test_start + SEQ_LEN  # 877
    test_data_len = len(df_raw) - test_start  # 315
    tot_len = test_data_len - SEQ_LEN - PRED_LEN + 1  # 172

    cols_data = [c for c in df_raw.columns if c != 'date']
    df_data = df_raw[cols_data].copy()
    n_channels = len(cols_data)

    scaler = StandardScaler()
    scaler.fit(df_data.iloc[:num_train].values)
    data_scaled = scaler.transform(df_data.values)

    n_windows = tot_len if args_cli.full else min(20, tot_len)
    if not args_cli.full:
        print('使用前 {} 个窗口（加 --full 跑全部 {} 个窗口）'.format(n_windows, tot_len))

    pred_scaled = np.zeros((tot_len, PRED_LEN, n_channels))
    true_scaled = np.zeros((tot_len, PRED_LEN, n_channels))
    # 未跑的窗口用 nan，后面只对有效窗口算指标
    pred_scaled[:] = np.nan
    true_scaled[:] = np.nan

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print('请先安装: pip install statsmodels')
        return

    print('ARIMA 滚动预测（每窗口用历史拟合，预测 48 步）...')
    for ch in range(n_channels):
        series = data_scaled[:, ch].astype(np.float64)
        history = series[:forecast_start].tolist()
        for w in range(n_windows):
            try:
                model = ARIMA(history, order=(1, 1, 1))
                res = model.fit()
                f = res.forecast(steps=PRED_LEN)
                pred_scaled[w, :, ch] = np.asarray(f)
            except Exception:
                try:
                    model = ARIMA(history, order=(1, 0, 1))
                    res = model.fit()
                    f = res.forecast(steps=PRED_LEN)
                    pred_scaled[w, :, ch] = np.asarray(f)
                except Exception:
                    pred_scaled[w, :, ch] = np.nan
            start = forecast_start + w
            end = start + PRED_LEN
            true_scaled[w, :, ch] = series[start:end]
            if end <= len(series):
                history.extend(series[start:end].tolist())
            else:
                break

    # 只对有效窗口（无 nan）算指标
    valid = ~np.isnan(pred_scaled[:, 0, 0])
    n_valid = int(np.sum(valid))

    mae_s = np.nanmean(np.abs(pred_scaled - true_scaled))
    rmse_s = np.sqrt(np.nanmean((pred_scaled - true_scaled) ** 2))
    mae_per_s = np.nanmean(np.abs(pred_scaled - true_scaled), axis=(0, 1))
    rmse_per_s = np.sqrt(np.nanmean((pred_scaled - true_scaled) ** 2, axis=(0, 1)))

    # 逆变换到原始量纲（仅对有效窗口）
    pred_inv = np.full_like(pred_scaled, np.nan)
    true_inv = np.full_like(true_scaled, np.nan)
    if n_valid > 0:
        pred_inv[valid] = scaler.inverse_transform(pred_scaled[valid].reshape(-1, n_channels)).reshape(n_valid, PRED_LEN, n_channels)
        true_inv[valid] = scaler.inverse_transform(true_scaled[valid].reshape(-1, n_channels)).reshape(n_valid, PRED_LEN, n_channels)

    mae_inv = np.nanmean(np.abs(pred_inv - true_inv))
    rmse_inv = np.sqrt(np.nanmean((pred_inv - true_inv) ** 2))
    mae_per_inv = np.nanmean(np.abs(pred_inv - true_inv), axis=(0, 1))
    rmse_per_inv = np.sqrt(np.nanmean((pred_inv - true_inv) ** 2, axis=(0, 1)))

    print()
    print('=' * 50)
    print('ARIMA 基线（order=(1,1,1)，滚动拟合，有效窗口数={})'.format(n_valid))
    print('=' * 50)
    print('【不逆变换，标准化空间】')
    print('Test 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(mae_s, rmse_s))
    print('各指标:')
    for i in range(n_channels):
        print('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(cols_data[i], mae_per_s[i], rmse_per_s[i]))
    print('-' * 50)
    print('【逆变换到原始量纲】')
    print('Test 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(mae_inv, rmse_inv))
    print('各指标:')
    for i in range(n_channels):
        print('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(cols_data[i], mae_per_inv[i], rmse_per_inv[i]))
    print('=' * 50)


if __name__ == '__main__':
    main()
