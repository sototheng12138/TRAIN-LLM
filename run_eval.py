"""
加载训练好的 checkpoint，在测试集上跑一遍，计算 MAE 和 RMSE。
用法：参数需与训练时一致（与 Iron.sh 一致），这样才能找到正确的 checkpoint 路径。
单卡即可：python run_eval.py 或 bash scripts/Eval_Iron.sh
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content
from utils.railway_semantic import generate_railway_semantic, generate_prototype_reasoning


def get_parser():
    p = argparse.ArgumentParser(description='Eval: load checkpoint and compute MAE/RMSE on test set')
    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--model_id', type=str, default='Iron_96_48')
    p.add_argument('--model_comment', type=str, default='iron')
    p.add_argument('--model', type=str, default='TimeLLM')
    p.add_argument('--data', type=str, default='custom')
    p.add_argument('--root_path', type=str, default='./dataset/')
    p.add_argument('--data_path', type=str, default='2023_2025_Iron_data.csv')
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='d')
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=48)
    p.add_argument('--checkpoints', type=str, default='./checkpoints/')
    p.add_argument('--enc_in', type=int, default=4)
    p.add_argument('--dec_in', type=int, default=4)
    p.add_argument('--c_out', type=int, default=4)
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--factor', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--activation', type=str, default='gelu', help='activation (for Autoformer etc.)')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--des', type=str, default='Iron_Ore_Transport_Exp')
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--use_multiscale_patch', action='store_true', help='match training: multi-scale patch (ablation)')
    p.add_argument('--no_revin', action='store_true', help='match training: ablation without RevIN')
    p.add_argument('--llm_model', type=str, default='LLAMA')
    p.add_argument('--llm_dim', type=int, default=4096)
    p.add_argument('--llm_layers', type=int, default=32)
    p.add_argument('--prompt_domain', type=int, default=0)
    p.add_argument('--prompt_type', type=str, default='full', choices=['full', 'short'], help='match training: full or short (ablation)')
    p.add_argument('--ablate_reprogramming', action='store_true', help='match training: ablation without reprogramming layer (linear projection only)')
    p.add_argument('--ablate_prompt', action='store_true', help='match training: ablation without prompt (reprogrammed patches only as LLM input)')
    p.add_argument('--ablate_prompt_description', action='store_true', help='match training: ablation without dataset description in prompt')
    p.add_argument('--ablate_prompt_task', action='store_true', help='match training: ablation without task instruction in prompt')
    p.add_argument('--ablate_prompt_stats', action='store_true', help='match training: ablation without input statistics in prompt')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seasonal_patterns', type=str, default='Monthly')
    p.add_argument('--percent', type=int, default=100)
    p.add_argument('--itr', type=int, default=1, help='experiment index, must match training (0-based for first run)')
    p.add_argument('--device', type=str, default='', help='device: cuda, cpu, or empty for auto (cuda if available)')
    p.add_argument('--eval_batch_size', type=int, default=8, help='batch size for eval, smaller to avoid OOM')
    p.add_argument('--moving_avg', type=int, default=25, help='DLinear moving average window')
    p.add_argument('--output_attention', action='store_true', help='whether to output attention (for Autoformer etc.)')
    p.add_argument('--multivariate', action='store_true', help='custom data: return (seq_len, enc_in) per sample; must match training when eval Iron_multivariate checkpoint')
    p.add_argument('--channel_mixing', action='store_true', help='model has channel mixing layer; must match training when eval Iron_multivariate checkpoint')
    p.add_argument('--save_pred_true', action='store_true', help='save pred and true to checkpoint dir for confusion matrix script')
    p.add_argument('--weight_decay', type=float, default=0.0, help='match training: if trained with weight_decay>0, pass same value to resolve checkpoint path')
    p.add_argument('--output_aux_semantic', action='store_true', help='append 辅助任务推断结果 (是否有发运) to semantic_output.txt (need checkpoint trained with use_aux_loss)')
    p.add_argument('--aux_confidence_threshold', type=float, default=0.5, help='when P(有发运)<this, treat whole-window pred as 0 (业务: 辅助头说无发运→系统直接给0). 0=disable. 需同时开启 output_aux_semantic')
    p.add_argument('--aux_smooth_kernel', type=int, default=0, help='if >0, smooth aux_probs with 1D avg_pool (kernel_size=this, stride=1, padding=1) before gating; 0=no smooth. Used by Eval_Iron_SmoothAux.sh')
    p.add_argument('--zero_threshold', type=float, default=0.0, help='after inverse_transform, set pred to 0 where |pred|<threshold. 0=disable. Use positive value (e.g. 1e-4, 0.01) or -1 for auto (scale from train std). Business: 先有无再多少')
    return p


def build_setting(args, ii=0):
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
        args.factor, args.embed, args.des, ii)


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'TimeLLM':
        args.content = load_content(args)
        if getattr(args, 'output_aux_semantic', False):
            args.use_aux_loss = True
    else:
        args.content = ''

    # 评估时用较小 batch 降低显存
    args.batch_size = getattr(args, 'eval_batch_size', args.batch_size)
    test_set, _ = data_provider(args, 'test')
    # 评估时 drop_last=False，保证拿到全部样本，便于按通道重组为 4 个指标
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    if args.model == 'TimeLLM':
        model = TimeLLM.Model(args).float()
    elif args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    else:
        model = DLinear.Model(args).float()

    setting = build_setting(args, ii=args.itr - 1)
    ckpt_dir = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    if getattr(args, 'ablate_reprogramming', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_reprogram'
    if getattr(args, 'ablate_prompt', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt'
    elif getattr(args, 'ablate_prompt_description', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt_desc'
    elif getattr(args, 'ablate_prompt_task', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt_task'
    elif getattr(args, 'ablate_prompt_stats', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt_stats'
    if getattr(args, 'llm_layers', 32) == 8:
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_llm8'
    if getattr(args, 'dropout', 0.1) == 0.15:
        ckpt_dir = ckpt_dir.rstrip('/') + '_dropout015'
    elif getattr(args, 'dropout', 0.1) == 0.05:
        ckpt_dir = ckpt_dir.rstrip('/') + '_dropout005'
    if getattr(args, 'weight_decay', 0.0) > 0:
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_weight_decay'
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint')

    if not os.path.exists(ckpt_path):
        print('Checkpoint not found:', ckpt_path)
        print('Make sure task_name, model_id, model_comment, data, features, seq_len, etc. match your training (e.g. Iron.sh).')
        return

    print('Loading checkpoint:', os.path.abspath(ckpt_path))
    state = torch.load(ckpt_path, map_location='cpu')
    # 兼容不同版本的模型（是否包含 aux_loss 相关参数），评估阶段放宽 strict 要求
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print('Warning: missing keys when loading checkpoint (ignored):', missing)
    if unexpected:
        print('Warning: unexpected keys when loading checkpoint (ignored):', unexpected)
    try:
        model = model.to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print('CUDA OOM, falling back to CPU (slower).')
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise
    model.eval()

    f_dim = -1 if args.features == 'MS' else 0
    all_pred, all_true = [], []
    all_aux_logits_list = []  # 每 batch 的辅助头 logits，用于辅助置信度置零
    first_batch_reprogramming_attn = None
    first_batch_aux_logits = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

            need_reprogramming_attn = (
                batch_idx == 0 and args.model == 'TimeLLM' and not getattr(args, 'ablate_reprogramming', False)
            )
            need_aux_semantic = (
                args.model == 'TimeLLM' and getattr(args, 'output_aux_semantic', False)
            )
            if need_reprogramming_attn or need_aux_semantic:
                out = model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    return_reprogramming_attention=need_reprogramming_attn,
                    return_aux_repr=need_aux_semantic,
                )
                if isinstance(out, tuple):
                    out, extra = out
                    if need_reprogramming_attn:
                        first_batch_reprogramming_attn = extra.get('reprogramming_attn')
                    if need_aux_semantic:
                        aux_this = extra.get('aux_logits')
                        if batch_idx == 0:
                            first_batch_aux_logits = aux_this
                        if aux_this is not None and 'has_shipment' in aux_this:
                            a = aux_this['has_shipment']
                            if hasattr(a, 'cpu'):
                                a = a.cpu().numpy()
                            all_aux_logits_list.append(np.asarray(a))
            else:
                out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            out = out[:, -args.pred_len:, f_dim:]
            true = batch_y[:, -args.pred_len:, f_dim:]
            out_np = out.cpu().numpy()
            true_np = true.cpu().numpy()
            # multivariate：模型输出 (B*4, pred_len, 4)，按窗口取对角得到 (B, pred_len, 4)
            if getattr(args, 'multivariate', False) and out_np.shape[0] == true_np.shape[0] * test_set.enc_in:
                B = true_np.shape[0]
                enc_in = test_set.enc_in
                out_agg = np.zeros((B, out_np.shape[1], enc_in), dtype=out_np.dtype)
                for b in range(B):
                    for c in range(enc_in):
                        out_agg[b, :, c] = out_np[b * enc_in + c, :, c]
                out_np = out_agg
            all_pred.append(out_np)
            all_true.append(true_np)

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    # Custom/Iron 独立通道：index = c*tot_len + t，重组为 (tot_len, pred_len, enc_in)
    n_channels = pred.shape[-1]
    if getattr(test_set, 'enc_in', None) and getattr(test_set, 'tot_len', None) and test_set.enc_in > 1 and pred.shape[-1] == 1:
        tot_len = test_set.tot_len
        enc_in = test_set.enc_in
        n_full = tot_len * enc_in
        # 正确顺序：先按通道再按窗口，所以用 (enc_in, tot_len, pred_len, 1) 再 transpose -> (tot_len, pred_len, enc_in)
        pred = pred[:n_full].reshape(enc_in, tot_len, args.pred_len, 1).transpose(1, 2, 0, 3).squeeze(-1)
        true = true[:n_full].reshape(enc_in, tot_len, args.pred_len, 1).transpose(1, 2, 0, 3).squeeze(-1)
        n_channels = enc_in

    # 列名（逆变换前后共用）
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=0)
        col_names = list(df.columns[1:])  # 去掉 date
        if len(col_names) != n_channels:
            col_names = ['指标{}'.format(i + 1) for i in range(n_channels)]
    except Exception:
        col_names = ['指标{}'.format(i + 1) for i in range(n_channels)]

    # 先算标准化空间（不逆变换）的 MAE/RMSE
    mae_all_scaled = np.mean(np.abs(pred - true))
    rmse_all_scaled = np.sqrt(np.mean((pred - true) ** 2))
    mae_per_scaled = np.mean(np.abs(pred - true), axis=(0, 1))
    rmse_per_scaled = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)))

    lines = []
    lines.append('Checkpoint: {}'.format(ckpt_path))
    lines.append('-' * 50)
    lines.append('【不逆变换，标准化空间】')
    lines.append('Test 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(mae_all_scaled, rmse_all_scaled))
    lines.append('各指标:')
    for i in range(n_channels):
        lines.append('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(col_names[i], mae_per_scaled[i], rmse_per_scaled[i]))
    lines.append('-' * 50)
    for s in lines:
        print(s)

    # 逆变换到原始量纲后再算 MAE/RMSE
    if getattr(test_set, 'scale', False) and hasattr(test_set, 'inverse_transform'):
        pred_flat = pred.reshape(-1, n_channels)
        true_flat = true.reshape(-1, n_channels)
        pred = test_set.inverse_transform(pred_flat).reshape(pred.shape)
        true = test_set.inverse_transform(true_flat).reshape(true.shape)

    # 辅助头置信度置零：P(有发运)<阈值 的窗口整窗置 0。业务上「辅助头说无发运→系统直接给 0」，避免调度员面对 0.5 吨不知是否派车
    aux_conf_thresh = getattr(args, 'aux_confidence_threshold', 0.5)
    n_windows_zeroed_by_aux = 0
    gate_mode_str = '通道对通道'
    if aux_conf_thresh > 0 and len(all_aux_logits_list) > 0:
        all_aux = np.concatenate(all_aux_logits_list, axis=0)
        logits = np.asarray(all_aux, dtype=np.float64)
        logits_max = logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits - logits_max) / np.exp(logits - logits_max).sum(axis=-1, keepdims=True)
        n_windows = pred.shape[0]
        # 点对点辅助头：all_aux (n_windows*n_channels, pred_len, 2) -> gate (n_windows, pred_len, n_channels)
        if pred.ndim == 3 and logits.ndim == 3 and logits.shape[1] == pred.shape[1]:
            # 点对点：[B*N, pred_len, 2] -> (n_windows, n_channels, pred_len, 2) -> P(有发运) -> gate (n_windows, pred_len, n_channels)
            n_win, pred_len_dim, n_ch = pred.shape[0], pred.shape[1], pred.shape[2]
            aux_4d = logits.reshape(n_win, n_ch, pred_len_dim, 2)
            prob_ship_3d = probs.reshape(n_win, n_ch, pred_len_dim, 2)[..., 1].astype(np.float32)  # (n_win, n_ch, pred_len)
            aux_probs_bln = np.transpose(prob_ship_3d, (0, 2, 1))  # (n_win, pred_len, n_ch) = [B, L, N]
            smooth_k = getattr(args, 'aux_smooth_kernel', 0)
            if smooth_k > 0:
                t = torch.from_numpy(aux_probs_bln).float()
                smooth_probs = F.avg_pool1d(t.permute(0, 2, 1), kernel_size=smooth_k, stride=1, padding=smooth_k // 2).permute(0, 2, 1)
                aux_probs_bln = smooth_probs.numpy()
            gate = (aux_probs_bln > aux_conf_thresh).astype(pred.dtype)  # (n_windows, pred_len, n_channels)
            pred = pred * gate
            n_windows_zeroed_by_aux = int(np.sum(gate == 0))
            for c in range(n_channels):
                pc = gate[:, :, c]
                n_below_c = int(np.sum(pc <= 0))
                print('  [辅助头 P(有发运) 通道{} 点对点] ≤{:.2f} 的(窗,步)数={}/{}'.format(
                    c, aux_conf_thresh, n_below_c, n_win * pred_len_dim))
            if smooth_k > 0:
                gate_mode_str = '点对点(平滑k={})'.format(smooth_k)
            else:
                gate_mode_str = '点对点'
            print('  ({}：gate [B, Seq_Len, N]，共 {} 个(窗,步,通)被置零)'.format(gate_mode_str, n_windows_zeroed_by_aux))
            first_window_aux_ship_prob = float(aux_probs_bln[0, 0, 0])
            first_window_zeroed_by_aux = np.any(gate[0, :, :] == 0)
        else:
            prob_ship = probs[:, 1]  # (total_samples,) 旧版整窗/整通道
            if pred.ndim == 3 and prob_ship.size == n_windows * n_channels:
                aux_2d = prob_ship.reshape(n_windows, n_channels).astype(pred.dtype)
                gate = (aux_2d > aux_conf_thresh).astype(pred.dtype)
                pred = pred * gate[:, None, :]
                n_windows_zeroed_by_aux = int(np.sum(gate == 0))
                for c in range(n_channels):
                    pc = aux_2d[:, c]
                    n_below_c = int(np.sum(pc <= aux_conf_thresh))
                    print('  [辅助头 P(有发运) 通道{}] min={:.4f}, max={:.4f}, mean={:.4f}, ≤{:.2f} 的(窗,通)数={}/{}'.format(
                        c, float(np.min(pc)), float(np.max(pc)), float(np.mean(pc)), aux_conf_thresh, n_below_c, n_windows))
                print('  (解耦裁决：通道对通道，同一窗口同通道内 pred_len 步共用同一 gate)')
                first_window_aux_ship_prob = float(aux_2d[0, 0])
                first_window_zeroed_by_aux = np.any(aux_2d[0, :] <= aux_conf_thresh)
            elif pred.ndim == 3 and prob_ship.size == n_windows:
                aux_1d = np.asarray(prob_ship, dtype=pred.dtype)
                gate = (aux_1d > aux_conf_thresh).astype(pred.dtype)
                pred = pred * gate[:, None, None]
                n_windows_zeroed_by_aux = int(np.sum(gate == 0))
                first_window_aux_ship_prob = float(aux_1d[0])
                first_window_zeroed_by_aux = first_window_aux_ship_prob <= aux_conf_thresh
            else:
                first_window_aux_ship_prob = None
                first_window_zeroed_by_aux = False
    else:
        first_window_aux_ship_prob = None
        first_window_zeroed_by_aux = False

    # 零阈值：接近 0 的预测置为真 0。业务上先有无发运再看发多少；仅评估阶段做，训练不截断以保留梯度
    zero_thresh = getattr(args, 'zero_threshold', 0.0)
    n_zeroed = 0
    if zero_thresh != 0:
        if zero_thresh == -1 or zero_thresh < 0:
            # auto：用训练集各通道 std 的最小值的 5% 作为阈值（量纲挂钩）；若仍过小则用预测值最小量级的 10%
            if getattr(test_set, 'scale', False) and hasattr(test_set, 'scaler') and hasattr(test_set.scaler, 'scale_'):
                std_per = np.asarray(test_set.scaler.scale_)
                zero_thresh = float(np.maximum(1e-10, 0.05 * np.min(std_per)))
            else:
                zero_thresh = 1e-4
            # 逆变换后 pred 在原始量纲，若 0.05*min(std) 仍很小（如<0.01），改用「预测绝对值 5% 分位」避免阈值过小导致零作用
            abs_pred = np.abs(pred)
            if zero_thresh < 0.01 and np.any(abs_pred > 0):
                pct = np.percentile(abs_pred[abs_pred > 0], 5)
                if not np.isnan(pct):
                    zero_thresh = max(zero_thresh, float(pct))
        if zero_thresh > 0:
            mask = np.abs(pred) < zero_thresh
            n_zeroed = int(np.sum(mask))
            pred = np.where(mask, 0.0, pred)

    mae_all = np.mean(np.abs(pred - true))
    rmse_all = np.sqrt(np.mean((pred - true) ** 2))
    mae_per = np.mean(np.abs(pred - true), axis=(0, 1))
    rmse_per = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)))

    lines.append('【逆变换到原始量纲】')
    if aux_conf_thresh > 0 and len(all_aux_logits_list) > 0:
        lines.append('  (C_aux 解耦裁决：preds = preds * (aux_probs > {:.2f})，{}，共 {} 个置零；在 MAE/零阈值之前执行)'.format(aux_conf_thresh, gate_mode_str, n_windows_zeroed_by_aux))
        print('  [C_aux 裁决] preds = preds * (aux_probs > {:.2f})，{}，{} 个置零（裁决在 MAE/零阈值之前已执行）'.format(aux_conf_thresh, gate_mode_str, n_windows_zeroed_by_aux))
        if n_windows_zeroed_by_aux == 0:
            print('  → 若为 0：无(窗口,通道)被置零；MAE/图中已是裁决后结果')
    if zero_thresh > 0:
        lines.append('  (已应用零阈值 |pred|<{:.4g} → 0，本批共 {} 个预测被置零；仅评估阶段)'.format(zero_thresh, n_zeroed))
        print('  [零阈值] threshold={:.4g}, 被置零元素数={}'.format(zero_thresh, n_zeroed))
        if n_zeroed == 0 and zero_thresh < 1000:
            hint = '  提示：逆变换后数据量纲较大，当前阈值过小导致无预测被置零；建议改用 --zero_threshold -1（自动）或更大数值（如 10000）'
            lines.append(hint)
            print(hint)
    lines.append('Test 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(mae_all, rmse_all))
    lines.append('各指标:')
    for i in range(n_channels):
        lines.append('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(col_names[i], mae_per[i], rmse_per[i]))
    # 各列量纲不同，逆变换后 MAE/RMSE 会差很多；打印相对误差 MAE/std、RMSE/std 说明标准化下表现是否接近
    if getattr(test_set, 'scale', False) and hasattr(test_set, 'scaler') and hasattr(test_set.scaler, 'scale_'):
        std_per = np.asarray(test_set.scaler.scale_)
        lines.append('各指标 MAE/训练集std（相对误差，越接近说明量纲差异导致逆变换MAE不同）:')
        for i in range(n_channels):
            rel = mae_per[i] / std_per[i] if std_per[i] > 0 else float('nan')
            lines.append('  {}  MAE/std = {:.4f}'.format(col_names[i], rel))
        lines.append('各指标 RMSE/训练集std（标准化RMSE，与MAE/std同理）:')
        for i in range(n_channels):
            rel_rmse = rmse_per[i] / std_per[i] if std_per[i] > 0 else float('nan')
            lines.append('  {}  RMSE/std = {:.4f}'.format(col_names[i], rel_rmse))
    lines.append('-' * 50)
    # 打印逆变换后的结果（上面已打印标准化空间部分）
    n_first = 6 + n_channels  # 第一段行数（Checkpoint + 分隔 + 标题 + Test整体 + "各指标" + n_channels行 + 分隔）
    for s in lines[n_first:]:
        print(s)
    # 保存完整结果到 checkpoint 目录，便于查找各消融结果
    eval_result_path = os.path.join(ckpt_dir, 'eval_result.txt')
    try:
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print('结果已保存:', eval_result_path)
    except Exception as e:
        print('保存结果文件失败:', e)
    print('-' * 50)

    # 可选：保存 pred/true 供混淆矩阵等后处理使用
    if getattr(args, 'save_pred_true', False):
        try:
            np.save(os.path.join(ckpt_dir, 'pred.npy'), pred)
            np.save(os.path.join(ckpt_dir, 'true.npy'), true)
            print('已保存 pred.npy / true.npy 到', ckpt_dir)
        except Exception as e:
            print('保存 pred/true 失败:', e)

    # 四通道预测图像：pred vs true，首窗口；中文显示 + 清晰优雅版
    if pred.ndim >= 3 and pred.shape[0] > 0 and pred.shape[2] >= n_channels:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # 中文字体：若图中中文为方框，请安装其一后重跑，如 Ubuntu: sudo apt install fonts-wqy-microhei fonts-noto-cjk
            plt.rcParams['font.sans-serif'] = [
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
                'SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans'
            ]
            plt.rcParams['axes.unicode_minus'] = False
            pred_len = pred.shape[1]
            n_plot = min(4, n_channels)
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
            axes = axes.flatten()
            t = np.arange(pred_len)
            # 配色：真实值深蓝实线，预测值橙红虚线，更易区分
            color_true, color_pred = '#1f77b4', '#d62728'
            for c in range(n_plot):
                ax = axes[c]
                ax.plot(t, true[0, :, c], color=color_true, linestyle='-', label='真实值', alpha=0.95, linewidth=2.2)
                ax.plot(t, pred[0, :, c], color=color_pred, linestyle='--', label='预测值', alpha=0.9, linewidth=1.5)
                ax.set_title(col_names[c] if c < len(col_names) else 'Ch{}'.format(c), fontsize=12, fontweight='medium')
                ax.set_xlabel('预测步', fontsize=10)
                ax.set_ylabel('发运量（原始量纲）', fontsize=10)
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                ax.grid(True, which='both', alpha=0.25, linestyle='-')
                ax.tick_params(axis='both', labelsize=9)
                ax.set_xlim(-0.5, pred_len - 0.5)
            plt.tight_layout()
            plot_png = os.path.join(ckpt_dir, 'pred_true_4channels.png')
            plot_svg = os.path.join(ckpt_dir, 'pred_true_4channels.svg')
            fig.savefig(plot_png, dpi=180, bbox_inches='tight', facecolor='white')
            fig.savefig(plot_svg, format='svg', bbox_inches='tight', facecolor='white')
            plt.close()
            # 保存绘图数据，之后改图时可直接加载重绘，无需重跑评估
            data_path = os.path.join(ckpt_dir, 'pred_true_4channels_data.npz')
            np.savez(
                data_path,
                pred_first=pred[0],
                true_first=true[0],
                pred_len=np.array(pred_len),
                col_names=np.array(col_names[:n_plot], dtype=object),
            )
            print('已保存四通道预测图:', plot_png, ',', plot_svg)
            print('已保存绘图数据:', data_path, '（改图可运行: python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir "' + ckpt_dir + '"）')
        except Exception as e:
            print('绘制预测图失败:', e)

    # 时间推演模块：历史 + 预测融合，生成铁路业务逻辑的语义解释（以第一个测试窗口为例）
    # 若有重编程层 attention，则结合文本原型生成「原型推理」段落
    if getattr(test_set, 'data_x', None) is not None and getattr(test_set, 'inverse_transform', None) and pred.shape[0] > 0:
        try:
            seq_len = getattr(args, 'seq_len', 96)
            hist_scaled = test_set.data_x[:seq_len]  # (seq_len, n_channels)
            hist_orig = test_set.inverse_transform(hist_scaled)
            pred_first = pred[0]  # (pred_len, n_channels)，已为原始量纲
            semantic = generate_railway_semantic(
                hist_orig,
                pred_first,
                col_names,
                pred_len=args.pred_len,
            )
            # 重编程层文本原型推理：结合首 batch 的 patch→原型 attention
            if first_batch_reprogramming_attn is not None:
                prototype_words_by_id = None
                for name in ('prototype_topk_words_decoded.json', 'prototype_topk_words.json'):
                    proto_path = os.path.join(ckpt_dir, 'prototype_export', name)
                    if os.path.exists(proto_path):
                        try:
                            with open(proto_path, 'r', encoding='utf-8') as f:
                                raw = json.load(f)
                            prototype_words_by_id = {int(k): v for k, v in raw.items()}
                            break
                        except Exception:
                            continue
                prototype_reasoning = generate_prototype_reasoning(
                    first_batch_reprogramming_attn,
                    prototype_words_by_id=prototype_words_by_id,
                    sample_idx=0,
                    top_k=5,
                )
                semantic = semantic + '\n\n' + prototype_reasoning
            # 辅助任务推断结果（确定性解码）：二分类「是否有发运」→ 更具体的模板文本（置信度 + 与数值预测对照 + 业务建议）
            if first_batch_aux_logits is not None and 'has_shipment' in first_batch_aux_logits:
                logits = first_batch_aux_logits['has_shipment']
                if hasattr(logits, 'cpu'):
                    logits = logits.cpu().numpy()
                logits = np.asarray(logits, dtype=np.float64)
                if logits.ndim == 3:
                    logits_0 = logits[0].mean(axis=0)  # 点对点 (pred_len, 2) -> 首样本窗内平均 (2,)
                else:
                    logits_0 = logits[0]
                probs = np.exp(logits_0 - logits_0.max()) / np.exp(logits_0 - logits_0.max()).sum()
                dispatch_pred = int(np.argmax(logits_0))
                conf_pct = int(round(100 * probs[dispatch_pred]))
                # 数值预测上本窗口是否“有发运”（存在明显非零）
                pred_arr = np.asarray(pred_first)
                pred_has_any = np.any(pred_arr > 1e-4) if pred_arr.size else False
                # 预测期内“有发运”的天数：按时间步，任一道有量即计为 1 天
                if pred_arr.ndim >= 2:
                    days_with_shipment = int(np.sum(np.any(pred_arr > 1e-4, axis=1)))
                else:
                    days_with_shipment = int(np.sum(pred_arr > 1e-4)) if pred_arr.size else 0
                pred_len_days = pred_arr.shape[0] if pred_arr.ndim >= 1 else 0
                day_desc = '{}天'.format(days_with_shipment) if pred_has_any and pred_len_days else '若干天'
                if dispatch_pred == 1:
                    consis = '与数值预测结果（预测期内存在发运量）一致。' if pred_has_any else '数值预测显示预测期发运量较低，可结合上文趋势与品种关系综合研判。'
                    aux_text = (
                        '【辅助任务推断结果】基于联合训练得到的辅助分类头，模型以约 **{}%** 的置信度判定：本预测窗口内**存在发运活动**——'
                        '预测期内存在至少{}的发运安排，建议关注车皮与装卸能力。{} '
                        '该判定由骨干隐表示经二分类头直接输出，属基于模板的确定性解码，无大模型幻觉。'
                    ).format(conf_pct, day_desc, consis)
                else:
                    consis = '与数值预测结果（预测期整体接近零）一致。' if not pred_has_any else '可与数值预测及上文发运连续性、品种关系综合核对。'
                    aux_text = (
                        '【辅助任务推断结果】基于联合训练得到的辅助分类头，模型以约 **{}%** 的置信度判定：本预测窗口内**无发运**——'
                        '预测期内整体无发运计划，可能对应检修、停运或计划空档，建议结合日历与计划核对。{} '
                        '该判定由骨干隐表示经二分类头直接输出，属基于模板的确定性解码，无大模型幻觉。'
                    ).format(conf_pct, consis)
                semantic = semantic + '\n\n' + aux_text
            # 当首窗口被「辅助头置信度」整窗置 0 时，单独写出「主预测+置信度→系统给0」，突出懂业务、可操作（ESWA 审稿关切）
            if first_window_zeroed_by_aux and first_window_aux_ship_prob is not None:
                conf_pct_ship = int(round(100 * first_window_aux_ship_prob))
                sys_zero_note = (
                    '\n\n【系统决策说明】主预测存在小值（接近噪声量级）时，辅助头给出「有发运」置信度仅 **{}%**（< 设定阈值 {}%），'
                    '系统已将该窗口预测整窗置 0，视为无发运/噪声。**调度员可直接采用：无需派车。**'
                ).format(conf_pct_ship, int(round(100 * aux_conf_thresh)))
                semantic = semantic + sys_zero_note
            print('【时间推演与铁路业务语义解释】（基于首窗口：历史 {} 天 + 预测 {} 天）'.format(seq_len, args.pred_len))
            print('-' * 50)
            print(semantic)
            print('-' * 50)
            # 同步写入文件，便于从「文本原型」角度查看评估输出
            try:
                semantic_path = os.path.join(ckpt_dir, 'semantic_output.txt')
                with open(semantic_path, 'w', encoding='utf-8') as f:
                    f.write('【时间推演与铁路业务语义解释】基于首窗口：历史 {} 天 + 预测 {} 天\n\n'.format(seq_len, args.pred_len))
                    f.write(semantic)
                print('语义解释已写入:', semantic_path)
            except Exception as e:
                print('写入 semantic_output.txt 失败:', e)
        except Exception as e:
            print('时间推演模块未执行: {}'.format(e))


if __name__ == '__main__':
    main()
