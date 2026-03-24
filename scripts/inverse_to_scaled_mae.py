#!/usr/bin/env python3
"""
从「逆变换到原始量纲」的 MAE/RMSE 反推「标准化空间」的 MAE/RMSE。
用法：用 Iron 数据集加载一次得到 train std，再对给定的逆变换结果做除以 std 的换算。
"""
import os
import sys
import numpy as np

# 与 run_eval 相同的数据集与划分，得到相同的 scaler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_factory import data_provider
from utils.tools import load_content

def get_iron_scaler():
    class Args:
        root_path = './dataset/'
        data_path = '2023_2025_Iron_data.csv'
        data = 'custom'
        features = 'M'
        seq_len = 96
        label_len = 48
        pred_len = 48
        enc_in = 4
        percent = 100
        embed = 'timeF'
        freq = 'h'
        model_comment = 'iron'
        content = ''
        batch_size = 32
        num_workers = 0
        seasonal_patterns = None
        target = 'OT'
    args = Args()
    test_set, _ = data_provider(args, 'test')
    if not getattr(test_set, 'scale', False) or not hasattr(test_set, 'scaler') or not hasattr(test_set.scaler, 'scale_'):
        raise RuntimeError('Dataset has no scaler.scale_')
    return np.asarray(test_set.scaler.scale_), getattr(test_set, 'enc_in', 4)

def inverse_to_scaled(mae_orig_per, rmse_orig_per, mae_orig_all, rmse_orig_all, std_per, col_names=None):
    """逆变换空间的 MAE/RMSE 除以各通道 std，得到标准化空间（与 eval_result 一致）。"""
    n = len(std_per)
    if col_names is None:
        col_names = ['铁矿石', '铁矿砂', '铁矿粉', '铁精矿粉'][:n]
    mae_s = np.array([mae_orig_per[i] / std_per[i] if std_per[i] > 0 else np.nan for i in range(n)])
    rmse_s = np.array([rmse_orig_per[i] / std_per[i] if std_per[i] > 0 else np.nan for i in range(n)])
    mae_all_s = np.nanmean(mae_s)
    rmse_all_s = np.sqrt(np.nanmean(rmse_s ** 2))
    return mae_all_s, rmse_all_s, mae_s, rmse_s, col_names

if __name__ == '__main__':
    # 你给的逆变换结果
    mae_orig_all = 754790.250000
    rmse_orig_all = 1203357.250000
    mae_orig_per = [
        323484.437500,   # 铁矿石
        1473781.000000,  # 铁矿砂
        8305.229492,     # 铁矿粉
        1213590.375000,  # 铁精矿粉
    ]
    rmse_orig_per = [
        641802.750000,
        1790391.500000,
        113332.351562,
        1470380.375000,
    ]
    col_names = ['铁矿石', '铁矿砂', '铁矿粉', '铁精矿粉']

    std_per, enc_in = get_iron_scaler()
    assert len(std_per) >= len(mae_orig_per), 'scaler channel mismatch'
    std_per = std_per[:len(mae_orig_per)]

    mae_all_s, rmse_all_s, mae_s, rmse_s, _ = inverse_to_scaled(
        mae_orig_per, rmse_orig_per, mae_orig_all, rmse_orig_all, std_per, col_names
    )

    print('【不逆变换，标准化空间】（由逆变换结果除以 train std 换算）')
    print('Test 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(mae_all_s, rmse_all_s))
    print('各指标:')
    for i in range(len(col_names)):
        print('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(col_names[i], mae_s[i], rmse_s[i]))
    print('各指标 MAE/训练集std（与上面各指标 MAE 一致）:')
    for i in range(len(col_names)):
        print('  {}  MAE/std = {:.4f}'.format(col_names[i], mae_s[i]))
