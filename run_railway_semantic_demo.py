# -*- coding: utf-8 -*-
"""
仅运行时间推演模块的演示：不加载预测模型，用测试集首窗口的「历史 + 真实未来」模拟「历史 + 预测」，
生成铁路业务语义解释。用于验证语义模块；正式评估时由 run_eval.py 在得到模型预测后自动调用。
"""
import os
import sys
import numpy as np

# 与 run_eval 相同的数据参数
ROOT_PATH = './dataset/'
DATA_PATH = '2023_2025_Iron_data.csv'
SEQ_LEN = 96
PRED_LEN = 48

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_provider.data_factory import data_provider
    from utils.railway_semantic import generate_railway_semantic

    class Args:
        root_path = ROOT_PATH
        data_path = DATA_PATH
        data = 'custom'
        features = 'M'
        seq_len = SEQ_LEN
        label_len = 48
        pred_len = PRED_LEN
        target = 'OT'
        embed = 'timeF'
        freq = 'd'
        percent = 100
        batch_size = 8
        num_workers = 0
        seasonal_patterns = 'Monthly'

    args = Args()
    test_set, _ = data_provider(args, 'test')

    if not hasattr(test_set, 'data_x') or not hasattr(test_set, 'inverse_transform'):
        print('当前数据集无 data_x / inverse_transform，无法运行演示。')
        return

    # 首窗口：历史 = data_x[0:seq_len]，未来 = data_y 中对应预测段（这里用真实值模拟“预测”）
    hist_scaled = test_set.data_x[:SEQ_LEN]
    # 预测段在 data_y 中对应 [seq_len : seq_len+pred_len]
    future_scaled = test_set.data_y[SEQ_LEN : SEQ_LEN + PRED_LEN]
    hist_orig = test_set.inverse_transform(hist_scaled)
    future_orig = test_set.inverse_transform(future_scaled)

    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT_PATH, DATA_PATH), nrows=0)
        col_names = list(df.columns[1:])
    except Exception:
        col_names = ['铁矿石', '铁矿砂', '铁矿粉', '铁精矿粉']

    print('【时间推演与铁路业务语义解释】演示（首窗口：历史 {} 天 + “预测” {} 天，此处用真实值模拟预测）'.format(SEQ_LEN, PRED_LEN))
    print('-' * 50)
    semantic = generate_railway_semantic(hist_orig, future_orig, col_names, pred_len=PRED_LEN)
    print(semantic)
    print('-' * 50)

if __name__ == '__main__':
    main()
