#!/usr/bin/env python3
"""
将 m5.csv（10 行 x 1941 列，每行一个序列/组，无表头）转为可读格式：日期 + 数值列，行=时间步。
- m5_4ch.csv：前 4 列，供主模型 (enc_in=4) zero-shot/few-shot/lightweight 用。
- m5_10ch.csv：全部 10 列，供 10 通道训练/评估，泛化结果覆盖 10 组。
"""
import os
import pandas as pd

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'dataset', 'm5.csv')

    df = pd.read_csv(src, sep='\t', header=None)
    assert df.shape[1] == 1941, f"Expected 1941 time steps (columns), got {df.shape[1]}"
    n_series = df.shape[0]  # 10 组
    data = df.values.T  # (1941, n_series)
    n_days = data.shape[0]
    dates = pd.date_range('2011-01-01', periods=n_days, freq='D')

    # 4 通道：与主模型 enc_in=4 一致
    out_4 = os.path.join(root, 'dataset', 'm5_4ch.csv')
    out_4ch = pd.DataFrame({'date': dates})
    for i in range(min(4, n_series)):
        out_4ch[f'series_{i}'] = data[:, i]
    out_4ch.to_csv(out_4, index=False)
    print(f"Written: {out_4}  shape={out_4ch.shape}  (date+4)")

    # 10 通道：全部 10 组，泛化结果完整
    out_10 = os.path.join(root, 'dataset', 'm5_10ch.csv')
    out_10ch = pd.DataFrame({'date': dates})
    for i in range(n_series):
        out_10ch[f'series_{i}'] = data[:, i]
    out_10ch.to_csv(out_10, index=False)
    print(f"Written: {out_10}  shape={out_10ch.shape}  (date+{n_series})")

    # 用同一 4 通道 checkpoint 覆盖 10 组：分 3 个 4 通道子集，eval 3 次再合并
    for name, cols in [
        ('m5_4ch_0_3', [0, 1, 2, 3]),
        ('m5_4ch_4_7', [4, 5, 6, 7]),
        ('m5_4ch_8_9', [8, 9, 8, 9]),  # 8,9 再重复一列凑 4 通道，评估时只取前 2 个指标
    ]:
        p = os.path.join(root, 'dataset', name + '.csv')
        d = pd.DataFrame({'date': dates})
        for i, c in enumerate(cols):
            d[f'series_{i}'] = data[:, c]
        d.to_csv(p, index=False)
        print(f"Written: {p}  (4ch subset for 10-group eval)")

if __name__ == '__main__':
    main()
