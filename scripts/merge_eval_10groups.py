#!/usr/bin/env python3
"""
合并 3 次 4 通道 eval 的结果为 10 组指标。
输入：eval_m5_0_3.txt（4 行）, eval_m5_4_7.txt（4 行）, eval_m5_8_9.txt（取前 2 行）
每行格式：  series_0  MAE = x.xx  RMSE = y.yy  或  指标1  MAE = ...
"""
import re
import sys
import os

def parse_metrics(path, max_rows=4):
    """只解析【逆变换到原始量纲】下「各指标:」后的那几行，返回 [(name, mae, rmse), ...]"""
    if not os.path.isfile(path):
        return []
    lines = open(path, encoding='utf-8').readlines()
    out = []
    idx_inv = next((i for i, l in enumerate(lines) if '【逆变换到原始量纲】' in l), -1)
    if idx_inv < 0:
        return []
    idx_start = next((i for i in range(idx_inv + 1, len(lines)) if lines[i].strip() == '各指标:'), -1)
    if idx_start < 0:
        return []
    for i in range(idx_start + 1, min(idx_start + 1 + max_rows, len(lines))):
        m = re.search(r'\s+(\S+)\s+MAE\s*=\s*([\d.e+-]+)\s+RMSE\s*=\s*([\d.e+-]+)', lines[i])
        if m:
            out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(root, 'checkpoints', 'long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_stage2_linear')
    f0 = os.path.join(base, 'eval_m5_0_3.txt')
    f1 = os.path.join(base, 'eval_m5_4_7.txt')
    f2 = os.path.join(base, 'eval_m5_8_9.txt')

    all_metrics = []
    for i, path in enumerate([f0, f1, f2]):
        rows = parse_metrics(path)
        if i == 2:
            rows = rows[:2]  # 8_9 只取前 2 个
        for r in rows:
            all_metrics.append(r)

    if len(all_metrics) != 10:
        print('Warn: expected 10 rows, got', len(all_metrics), file=sys.stderr)
    print('========== 10 组泛化结果（同一 4 通道 checkpoint，分 3 次 eval 合并）==========')
    for i, (name, mae, rmse) in enumerate(all_metrics):
        print('  series_{}  MAE = {:.6f}  RMSE = {:.6f}'.format(i, mae, rmse))
    out_path = os.path.join(base, 'eval_result_m5_10groups_4ch_ckpt.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('10 组泛化结果（4 通道 checkpoint 分 3 次 eval 合并）\n')
        for i, (name, mae, rmse) in enumerate(all_metrics):
            f.write('  series_{}  MAE = {:.6f}  RMSE = {:.6f}\n'.format(i, mae, rmse))
    print('Written:', out_path)

if __name__ == '__main__':
    main()
