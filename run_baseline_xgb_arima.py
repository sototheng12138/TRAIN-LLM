#!/usr/bin/env python3
"""
XGBoost / ARIMA / LSTM / Prophet 基线，与 Time-LLM 严格对齐，保证公平对比：

公平性设置（与 data_loader / run_eval 一致）：
- 数据划分：Train : Val : Test = 7 : 1 : 2，仅用前 70% 训练，最后 20% 测试
- 测试窗口：共 num_test - pred_len + 1 个（与主模型 test set 样本数一致）
- 标准化：各通道用训练集 StandardScaler fit，MAE/RMSE 为原始误差/训练集 std，与 eval_result.txt【不逆变换，标准化空间】同口径
- 默认使用全部测试窗口（--full）；加 --quick 时 ARIMA/Prophet 仅用前 30 窗以省时（不利于公平对比，仅作快速调试）

运行结束后将四种模型的指标写入 baseline_eval_result.txt，便于与主模型直接对比。

依赖: pip install pandas numpy scikit-learn statsmodels xgboost torch prophet
"""
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')
# Prophet 底层使用 Stan (cmdstanpy)，会刷屏输出 Chain start/done；降至 WARNING 避免刷屏
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# ---------- 配置（与 Dataset_Custom_Iron / run_eval 一致）----------
SEQ_LEN = 96
PRED_LEN = 48
ROOT = './dataset/'
DATA_PATH = '2023_2025_Iron_data.csv'
COLS = ['铁矿石', '铁矿砂', '铁矿粉', '铁精矿粉']
OUTPUT_TXT = 'baseline_eval_result.txt'  # 结果保存路径（项目根目录下）

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


# ---------- 数据 ----------
def load_data():
    path = os.path.join(ROOT, DATA_PATH)
    df = pd.read_csv(path)
    if '日期' in df.columns:
        df = df.rename(columns={'日期': 'date'})
    df = df.sort_values('date').reset_index(drop=True)
    return df


def create_windows(data, seq_len, pre_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pre_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pre_len])
    return np.array(X), np.array(y)


# ---------- ARIMA ----------
def run_arima_rolling(ts_train, ts_test, pre_len, train_std, max_windows=None):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return None, None, None, None, None
    n_test = len(ts_test)
    if n_test < pre_len:
        return None, None, None, None, None
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
        return None, None, None, None, None
    preds = np.array(preds)
    trues = np.array(trues)
    mae_orig = mean_absolute_error(trues, preds)
    rmse_orig = np.sqrt(mean_squared_error(trues, preds))
    mae_s = mae_orig / train_std if train_std > 0 else np.nan
    rmse_s = rmse_orig / train_std if train_std > 0 else np.nan
    return mae_s, rmse_s, len(preds), preds, trues


# ---------- Prophet ----------
def run_prophet_rolling(dates_train, ts_train, dates_test, ts_test, pre_len, train_std, max_windows=None):
    if not HAS_PROPHET:
        return None, None, None, None, None
    n_test = len(ts_test)
    if n_test < pre_len:
        return None, None, None, None, None
    n_windows = min(n_test - pre_len + 1, max_windows or n_test)
    preds, trues = [], []
    for i in range(n_windows):
        if i > 0:
            hist_dates = np.concatenate([dates_train, dates_test[:i]])
            hist_vals = np.concatenate([ts_train, ts_test[:i]])
        else:
            hist_dates = dates_train
            hist_vals = ts_train
        df_train = pd.DataFrame({'ds': pd.to_datetime(hist_dates), 'y': hist_vals})
        try:
            m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=True)
            m.fit(df_train)
            future = pd.DataFrame({'ds': pd.to_datetime(dates_test[i : i + pre_len])})
            f = m.predict(future)['yhat'].values
            preds.append(f)
            trues.append(ts_test[i : i + pre_len])
        except Exception:
            continue
    if len(preds) == 0:
        return None, None, None, None, None
    preds = np.array(preds)
    trues = np.array(trues)
    mae_orig = mean_absolute_error(trues, preds)
    rmse_orig = np.sqrt(mean_squared_error(trues, preds))
    mae_s = mae_orig / train_std if train_std > 0 else np.nan
    rmse_s = rmse_orig / train_std if train_std > 0 else np.nan
    return mae_s, rmse_s, len(preds), preds, trues


# ---------- LSTM ----------
def build_lstm(seq_len, pred_len, hidden=64):
    import torch
    import torch.nn as nn
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True, num_layers=2, dropout=0.1)
            self.linear = nn.Linear(hidden * seq_len, pred_len)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out.reshape(out.size(0), -1)
            return self.linear(out)
    return LSTMModel()


def run_lstm(X_train, y_train, X_test, y_test, train_std, epochs=50):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
    model = build_lstm(SEQ_LEN, PRED_LEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_train_t)
        loss = nn.functional.l1_loss(out, y_train_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).cpu().numpy()
    mae_orig = mean_absolute_error(y_test, pred)
    rmse_orig = np.sqrt(mean_squared_error(y_test, pred))
    mae_s = mae_orig / train_std if train_std > 0 else np.nan
    rmse_s = rmse_orig / train_std if train_std > 0 else np.nan
    return mae_s, rmse_s, pred, y_test


# ---------- 写入 txt（与 eval_result.txt 格式对齐，便于对比）----------
def write_baseline_result_txt(by_model_per_channel, by_model_overall, out_path, n_test_windows, arima_prophet_used_windows=None):
    """
    by_model_per_channel: dict, key=模型名, value=list of (mae, rmse) 每个通道一个
    by_model_overall: dict, key=模型名, value=(mae_overall, rmse_overall)
    n_test_windows: 测试窗口总数（与主模型一致）
    arima_prophet_used_windows: 若 ARIMA/Prophet 未用满全部窗口则注明，如 30
    """
    lines = []
    lines.append('Baseline 模型评估（与 Time-LLM 相同设置: 7:1:2, seq_len=96, pred_len=48）')
    lines.append('测试窗口总数: {}（与主模型 eval 一致；公平对比须使用全部窗口）'.format(n_test_windows))
    if arima_prophet_used_windows is not None and arima_prophet_used_windows < n_test_windows:
        lines.append('说明: ARIMA/Prophet 本次仅使用前 {} 个窗口（为公平对比请不加 --quick 使用全部 {} 个窗口）'.format(arima_prophet_used_windows, n_test_windows))
    lines.append('标准化 MAE/RMSE = 原始误差 / 训练集 std，与 checkpoints/.../eval_result.txt 中【不逆变换，标准化空间】可直接对比')
    lines.append('-' * 50)
    lines.append('【标准化空间，与主模型 eval_result.txt 可直接对比】')
    lines.append('')

    for name in ['XGBoost', 'ARIMA', 'LSTM', 'Prophet']:
        pairs = by_model_per_channel.get(name, [])
        overall = by_model_overall.get(name)
        if not pairs or overall is None:
            lines.append('模型: {}  (未运行或依赖未安装)'.format(name))
            lines.append('')
            continue
        mae_all, rmse_all = overall
        lines.append('模型: {}'.format(name))
        lines.append('Test 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(mae_all, rmse_all))
        lines.append('各指标:')
        for i, col in enumerate(COLS):
            if i < len(pairs):
                mae_s, rmse_s = pairs[i]
                lines.append('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(col, mae_s, rmse_s))
        lines.append('')
        lines.append('-' * 50)

    lines.append('')
    lines.append('主模型 Time-LLM (iron) 参考:')
    lines.append('Test 整体  MAE = 0.774738  RMSE = 1.237126')
    lines.append('（详见 checkpoints/long_term_forecast_Iron_96_48_TimeLLM_..._0-iron/eval_result.txt）')
    lines.append('-' * 50)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


def main():
    parser = argparse.ArgumentParser(description='基线模型评估，与 Time-LLM 公平对比')
    parser.add_argument('--quick', action='store_true', help='ARIMA/Prophet 仅用前 30 个测试窗口以省时（不利于公平对比）；默认使用全部窗口')
    args = parser.parse_args()
    # 公平对比：默认使用全部测试窗口；--quick 时 ARIMA/Prophet 仅用 30 窗
    use_full_windows = not args.quick
    max_windows = None if use_full_windows else 30

    df = load_data()
    total_len = len(df)
    num_train = int(total_len * 0.7)
    num_test = int(total_len * 0.2)
    num_vali = total_len - num_train - num_test
    test_start = total_len - num_test
    n_test_windows = num_test - PRED_LEN + 1  # 与 data_loader test set 样本数一致
    dates = pd.to_datetime(df['date']).values

    # 按模型名 -> 各通道 (mae, rmse)；以及各模型整体 (mae, rmse)
    by_model_per_channel = {'XGBoost': [], 'ARIMA': [], 'LSTM': [], 'Prophet': []}
    by_model_overall = {}
    results_table = []  # 用于打印的明细
    arima_prophet_used_windows = None if use_full_windows else 30
    # 首窗 pred/true 用于与 TimeLLM 同款四通道图（无截断机制，正常预测评估）
    first_true = np.zeros((PRED_LEN, len(COLS)), dtype=np.float64)
    first_pred = {n: np.zeros((PRED_LEN, len(COLS)), dtype=np.float64) for n in ['XGBoost', 'ARIMA', 'LSTM', 'Prophet']}

    print('=' * 60)
    print(' 与 Time-LLM 对齐的基线 (7:1:2, 仅 70% 训练, seq=96, pred=48)')
    print('=' * 60)
    print(' 总长度={}, 训练={}, 验证={}, 测试={}, 测试窗口数={}'.format(total_len, num_train, num_vali, num_test, n_test_windows))
    if not use_full_windows:
        print(' [注意] 已加 --quick：ARIMA/Prophet 仅用前 30 窗，与主模型非同一批窗口，对比不完全公平；去掉 --quick 可跑全部 {} 窗'.format(n_test_windows))
    if not HAS_XGB:
        print(' [提示] 未安装 xgboost，XGBoost 已跳过。安装: pip install xgboost')
    if not HAS_PROPHET:
        print(' [提示] 未安装 prophet，Prophet 已跳过。安装: pip install prophet')
    print()

    for col in COLS:
        ts = df[col].astype(float).values
        train_ts = ts[:num_train]
        test_ts = ts[test_start:]
        dates_train = dates[:num_train]
        dates_test = dates[test_start:]

        scaler = StandardScaler()
        scaler.fit(train_ts.reshape(-1, 1))
        train_std = np.sqrt(scaler.var_[0])

        context = ts[test_start - SEQ_LEN :]
        X_train, y_train = create_windows(train_ts, SEQ_LEN, PRED_LEN)
        X_test, y_test = create_windows(context, SEQ_LEN, PRED_LEN)
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        c = COLS.index(col)
        first_true[:, c] = y_test[0]

        # XGBoost
        if HAS_XGB:
            base = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            multi = MultiOutputRegressor(base)
            multi.fit(X_train, y_train)
            xgb_pred = multi.predict(X_test)
            mae_s = mean_absolute_error(y_test, xgb_pred) / train_std
            rmse_s = np.sqrt(mean_squared_error(y_test, xgb_pred)) / train_std
            by_model_per_channel['XGBoost'].append((mae_s, rmse_s))
            results_table.append({'指标': col, '模型': 'XGBoost', '标准化 MAE': mae_s, '标准化 RMSE': rmse_s})
            first_pred['XGBoost'][:, c] = xgb_pred[0]

        # ARIMA（与主模型公平：同一批测试窗口，默认全部）
        arima_out = run_arima_rolling(train_ts, test_ts, PRED_LEN, train_std, max_windows=max_windows)
        if arima_out[0] is not None:
            arima_mae_s, arima_rmse_s, n_win, arima_preds, arima_trues = arima_out
            by_model_per_channel['ARIMA'].append((arima_mae_s, arima_rmse_s))
            results_table.append({'指标': col, '模型': 'ARIMA', '标准化 MAE': arima_mae_s, '标准化 RMSE': arima_rmse_s})
            if len(arima_preds) > 0:
                first_pred['ARIMA'][:, c] = arima_preds[0]
            if col == COLS[0]:
                print('  ARIMA 使用 {} 个测试窗口/通道'.format(n_win))

        # LSTM（始终用全部测试窗口，与主模型一致）
        lstm_out = run_lstm(X_train, y_train, X_test, y_test, train_std)
        if lstm_out[0] is not None:
            lstm_mae_s, lstm_rmse_s, lstm_pred, _ = lstm_out
            by_model_per_channel['LSTM'].append((lstm_mae_s, lstm_rmse_s))
            results_table.append({'指标': col, '模型': 'LSTM', '标准化 MAE': lstm_mae_s, '标准化 RMSE': lstm_rmse_s})
            first_pred['LSTM'][:, c] = lstm_pred[0]

        # Prophet（与主模型公平：同一批测试窗口，默认全部）
        p_out = run_prophet_rolling(
            dates_train, train_ts, dates_test, test_ts, PRED_LEN, train_std, max_windows=max_windows
        )
        if p_out[0] is not None:
            p_mae_s, p_rmse_s, n_p, p_preds, p_trues = p_out
            by_model_per_channel['Prophet'].append((p_mae_s, p_rmse_s))
            results_table.append({'指标': col, '模型': 'Prophet', '标准化 MAE': p_mae_s, '标准化 RMSE': p_rmse_s})
            if len(p_preds) > 0:
                first_pred['Prophet'][:, c] = p_preds[0]
            if col == COLS[0]:
                print('  Prophet 使用 {} 个测试窗口/通道'.format(n_p))

    # 计算各模型整体（四通道平均方式与 eval 一致）
    for name in ['XGBoost', 'ARIMA', 'LSTM', 'Prophet']:
        pairs = by_model_per_channel[name]
        if not pairs:
            by_model_overall[name] = None
            continue
        mae_list = [p[0] for p in pairs]
        rmse_list = [p[1] for p in pairs]
        mae_all = np.mean(mae_list)
        rmse_all = np.sqrt(np.mean(np.array(rmse_list) ** 2))
        by_model_overall[name] = (mae_all, rmse_all)

    # 终端打印：明细表
    print()
    print('  指标      模型      标准化 MAE     标准化 RMSE')
    for row in results_table:
        print(' {} {:10s} {:.6e} {:.6e}'.format(
            row['指标'], row['模型'], row['标准化 MAE'], row['标准化 RMSE']))

    print()
    print('【与 eval_result.txt 可比】')
    for name in ['XGBoost', 'ARIMA', 'LSTM', 'Prophet']:
        o = by_model_overall.get(name)
        if o is None:
            continue
        print('  {:10s} 整体 标准化 MAE = {:.6f}  RMSE = {:.6f}'.format(name, o[0], o[1]))
    print()
    print(' 主模型 iron 标准化: MAE ≈ 0.7747, RMSE ≈ 1.2371（见 checkpoints/...iron/eval_result.txt）')
    print('=' * 60)

    # 写入 txt 文件（与主模型 eval_result.txt 格式对齐）
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_TXT)
    write_baseline_result_txt(by_model_per_channel, by_model_overall, out_path, n_test_windows, arima_prophet_used_windows)
    print(' 结果已写入: {}'.format(out_path))

    # 为各基线保存首窗 pred/true，便于用与 TimeLLM 同款脚本出图（无截断，正常预测）
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'baselines')
    for name in ['XGBoost', 'ARIMA', 'LSTM', 'Prophet']:
        if by_model_per_channel.get(name):
            ckpt_dir = os.path.join(base_dir, '{}_Iron_96_48'.format(name))
            os.makedirs(ckpt_dir, exist_ok=True)
            npz_path = os.path.join(ckpt_dir, 'pred_true_4channels_data.npz')
            np.savez(
                npz_path,
                pred_first=first_pred[name],
                true_first=first_true,
                pred_len=np.array(PRED_LEN),
                col_names=np.array(COLS, dtype=object),
            )
            print(' 基线 {} 绘图数据已保存: {}'.format(name, npz_path))
            print('    出图: python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir "{}"'.format(ckpt_dir))


if __name__ == '__main__':
    main()
