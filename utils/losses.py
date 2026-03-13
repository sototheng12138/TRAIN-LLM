# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(mase_loss, self).__init__()
        self.eps = eps  # 分母下限，避免 insample 几乎为常数时 1/masep 爆炸，导致 train loss 远高于 vali

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masep = masep.clamp(min=self.eps)  # 零膨胀/近常数序列时分母过小会导致 train loss 异常高
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class MASE_loss_multivariate(nn.Module):
    """MASE for (B,T,N) insample and (B,L,N) forecast/target; uses mase_loss per channel and averages."""
    def __init__(self, freq=1):
        super().__init__()
        self.mase = mase_loss()
        self.freq = freq

    def forward(self, insample: t.Tensor, forecast: t.Tensor, target: t.Tensor, mask: t.Tensor = None) -> t.Tensor:
        # insample (B, T, N), forecast/target (B, L, N)
        if mask is None:
            mask = t.ones_like(target)
        B, T, N = insample.shape
        _, L, _ = forecast.shape
        losses = []
        for c in range(N):
            # (B, T), (B, L), (B, L), (B, L)
            l = self.mase(
                insample[:, :, c],
                self.freq,
                forecast[:, :, c],
                target[:, :, c],
                mask[:, :, c],
            )
            losses.append(l)
        return t.stack(losses).mean()


class ZeroInflatedLoss(nn.Module):
    def __init__(self, zero_weight=2.0):
        super().__init__()
        self.zero_weight = zero_weight # 增加 0 值的惩罚权重

    def forward(self, forecast, target):
        mse = (forecast - target) ** 2
        # 当目标是 0 时，加大惩罚，强迫模型“学会”预测 0
        weight = t.where(target == 0, t.tensor(self.zero_weight).to(target.device), t.tensor(1.0).to(target.device))
        return t.mean(mse * weight)


class JointMaskedMSEAuxBCE(nn.Module):
    """
    联合损失：数值头纯 Masked MSE + 辅助头 BCE。训练时两路分道扬镳，绝不合并：
    - output_num → 只进 Masked MSE：有货日（target>0）往死里拟合吨数，没货日（target=0）不扣分、可放飞（毛刺/负数都行）。
    - output_aux → 只进 BCE：用 0/1 标签逼置信度在没货日压向 0、有货日推向 1。
    支持点对点辅助头：preds_aux (B*N, pred_len, 2)、targets_aux (B, pred_len, N)，逐点 BCE。
    推理时用 C_aux 裁决：C_aux < 阈值则该点置 0（点对点门控）。
    """
    def __init__(self, lambda_weight=1.0, eps=1e-8):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.eps = eps
        self.bce = nn.BCELoss()

    def forward(self, preds_num, preds_aux, targets_num, targets_aux, raw_targets_num=None):
        # preds_num / targets_num: 归一化空间（模型输入输出一致）；raw_targets_num 若提供则为原始量纲，用于 mask 与 aux 标签
        # 归一化陷阱：targets_num 若为 StandardScaler 后，真实 0 吨会变负值，(targets_num>0) 不等于「有货」；必须用 raw 判有/无货
        # preds_aux: (B*N, pred_len, 2) 点对点 logits，或 (B*N, 2) 旧版整窗
        # targets_aux: (B, pred_len, N) 点对点 0/1（必须由原始量纲生成），或 (B,) 整窗
        preds_num = preds_num.float()
        targets_num = targets_num.float()
        preds_aux = preds_aux.float()
        if preds_num.dim() == 3 and targets_num.dim() == 3 and preds_num.shape != targets_num.shape:
            B, pred_len, N = targets_num.shape
            if preds_num.numel() == B * N * pred_len:
                preds_num = preds_num.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
        # Mask：有货日才算 MSE；必须用原始量纲判「>0」，否则归一化后 0 吨变负、标签全乱
        if raw_targets_num is not None:
            raw_targets_num = raw_targets_num.float().to(device=preds_num.device)
            if raw_targets_num.shape != preds_num.shape and preds_num.dim() == 3 and targets_num.dim() == 3:
                B, pred_len, N = targets_num.shape
                if raw_targets_num.numel() == B * N * pred_len:
                    raw_targets_num = raw_targets_num.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
            mask = (raw_targets_num > 0).float()
        else:
            mask = (targets_num > 0).float()
        # 1) 数值头：纯 Masked MSE，没货日不扣分（绝对不加 loss_zero）
        squared_error = (preds_num - targets_num) ** 2
        masked_error = squared_error * mask
        loss_num = masked_error.sum() / (mask.sum() + self.eps)

        # 2) 辅助头 BCE：点对点 (B, pred_len, N) 或 整窗 (B,) 对齐
        if targets_aux.dim() == 3:
            # 点对点：preds_aux (B*N, pred_len, 2) -> probs (B, pred_len, N)
            B, pred_len, N = targets_aux.shape
            probs = preds_aux.softmax(dim=-1)[..., 1]  # (B*N, pred_len)
            probs = probs.view(B, N, pred_len).permute(0, 2, 1)  # (B, pred_len, N)
            targets_aux_float = targets_aux.float()
            loss_aux = self.bce(probs, targets_aux_float)
        else:
            if preds_aux.dim() >= 2 and preds_aux.shape[-1] == 2:
                probs = preds_aux.softmax(dim=-1)[..., 1]
            else:
                probs = preds_aux.clamp(1e-6, 1.0 - 1e-6)
            probs = probs.view(-1)
            targets_aux_float = targets_aux.float().view(-1)
            if probs.numel() != targets_aux_float.numel() and targets_aux_float.numel() > 0 and probs.numel() % targets_aux_float.numel() == 0:
                targets_aux_float = targets_aux_float.repeat_interleave(probs.numel() // targets_aux_float.numel())
            elif probs.numel() != targets_aux_float.numel():
                min_len = min(probs.numel(), targets_aux_float.numel())
                probs = probs[:min_len]
                targets_aux_float = targets_aux_float[:min_len]
            loss_aux = self.bce(probs, targets_aux_float)
        total = loss_num + self.lambda_weight * loss_aux
        return total