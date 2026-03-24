# -*- coding: utf-8 -*-
"""
时间推演模块：将历史序列与预测序列融合，推演趋势与品种关系，生成铁路业务逻辑的语义解释。
支持结合重编程层文本原型的激活权重，生成「原型推理」段落。
输入：历史多通道序列 (T_hist, C)、预测多通道序列 (T_pred, C)，均为原始量纲。
输出：一段面向铁路调度的中文语义解释。
"""
import numpy as np
from typing import List, Optional, Dict, Any, Union


def _fmt(x: float) -> str:
    """将大数格式化为万/百万便于阅读"""
    if abs(x) >= 1e6:
        return '{:.1f}万'.format(x / 1e4)
    if abs(x) >= 1e4:
        return '{:.1f}万'.format(x / 1e4)
    return '{:.0f}'.format(x)


def _trend_desc(mean_prev: float, mean_curr: float, name: str, thresh_ratio: float = 0.1) -> Optional[str]:
    """根据前后均值判断趋势并返回一句描述。"""
    if mean_prev <= 0 and mean_curr <= 0:
        return None
    if mean_prev <= 0:
        return '{}在预测期内由无发运转为有发运，需关注车皮与装卸安排。'.format(name)
    change = (mean_curr - mean_prev) / mean_prev
    if change >= thresh_ratio:
        return '{}预测期日均较历史近期上升约{:.0%}，建议预留运力。'.format(name, change)
    if change <= -thresh_ratio:
        return '{}预测期日均较历史近期下降约{:.0%}，可能与检修或计划调整有关。'.format(name, -change)
    return '{}预测期与历史近期水平相当，走势平稳。'.format(name)


def _zeros_ratio(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """沿 axis 统计零占比（每行或每列中 0 的占比）。这里用于每列（每个品种）的零占比。"""
    if arr.size == 0:
        return np.zeros(arr.shape[1] if arr.ndim > 1 else 1)
    if axis == 0 and arr.ndim == 2:
        return (np.abs(arr) < 1e-6).astype(float).sum(axis=0) / arr.shape[0]
    return np.zeros(1)


def _cross_channel_pattern(pred: np.ndarray, col_names: List[str], thresh_corr: float = 0.3) -> List[str]:
    """根据预测期各品种非零日是否重叠，生成品种间替代/并发的语义。"""
    sentences = []
    n = pred.shape[1]
    if n < 2 or pred.shape[0] < 2:
        return sentences
    # 二值：当日是否有发运（>小阈值视为有）
    active = (pred > 1e4).astype(float)
    # 两两相关系数
    for i in range(n):
        for j in range(i + 1, n):
            if active[:, i].sum() < 2 or active[:, j].sum() < 2:
                continue
            c = np.corrcoef(active[:, i], active[:, j])[0, 1]
            if np.isnan(c):
                continue
            if c >= thresh_corr:
                sentences.append('{}与{}在预测期内常同时有发运，存在联合排产或共线发运的可能。'.format(col_names[i], col_names[j]))
            elif c <= -thresh_corr:
                sentences.append('{}与{}在预测期内多呈“此有彼无”，存在运力竞争或计划交替，建议按日错峰安排车皮。'.format(col_names[i], col_names[j]))
    return sentences


def railway_semantic_from_fused(
    history: np.ndarray,
    pred: np.ndarray,
    col_names: List[str],
    pred_len: Optional[int] = None,
) -> str:
    """
    基于「历史 + 预测」融合后的序列，做时间推演并生成铁路业务语义解释。

    history: (T_hist, C) 历史段，原始量纲
    pred:    (T_pred, C) 预测段，原始量纲
    col_names: 各通道名称，如 ['铁矿石','铁矿砂','铁矿粉','铁精矿粉']
    pred_len: 可选，与 pred 第二维一致，用于说明“未来多少天”

    返回：一段完整的中文解释文本。
    """
    fused = np.vstack([history, pred])
    T_hist, T_pred = history.shape[0], pred.shape[0]
    C = pred.shape[1]
    if len(col_names) != C:
        col_names = ['指标{}'.format(i + 1) for i in range(C)]

    # 历史近期：取最后一段与预测等长，用于对比
    hist_tail_len = min(T_pred, T_hist)
    hist_tail = history[-hist_tail_len:] if hist_tail_len else history

    parts = []

    # 1) 各品种趋势：预测期均值 vs 历史近期均值
    parts.append('【趋势】')
    for c in range(C):
        mean_hist = float(np.mean(hist_tail[:, c])) if hist_tail.size else 0.0
        mean_pred = float(np.mean(pred[:, c]))
        s = _trend_desc(mean_hist, mean_pred, col_names[c])
        if s:
            parts.append('  ' + s)

    # 2) 预测期内零发运占比
    parts.append('【发运连续性】')
    zero_rat = _zeros_ratio(pred, axis=0)
    for c in range(C):
        if zero_rat[c] >= 0.5:
            parts.append('  {}在预测期内有超过一半日期无发运，呈批次/专列模式，与历史特征一致。'.format(col_names[c]))
        elif zero_rat[c] >= 0.2:
            parts.append('  {}在预测期内部分日期无发运，建议按日核对计划与车皮。'.format(col_names[c]))
        else:
            parts.append('  {}在预测期内发运较连续，可优先保障该品种排产稳定性。'.format(col_names[c]))

    # 3) 品种间关系（替代/并发）
    cross = _cross_channel_pattern(pred, col_names)
    if cross:
        parts.append('【品种关系】')
        for sent in cross:
            parts.append('  ' + sent)

    # 4) 预测期总量与主导品种
    parts.append('【预测期总体】')
    total_pred = float(np.sum(pred))
    if total_pred > 0:
        parts.append('  预测期{}天合计发运约{}吨。'.format(T_pred, _fmt(total_pred)))
        per_ch = np.sum(pred, axis=0)
        per_ch_pct = per_ch / total_pred
        order = np.argsort(-per_ch_pct)
        main = [col_names[order[0]]]
        if per_ch_pct[order[1]] >= 0.2:
            main.append(col_names[order[1]])
        parts.append('  预测期内以{}为主，建议据此预留车皮与装卸能力。'.format('、'.join(main)))
    else:
        parts.append('  预测期内合计发运接近零，请结合计划与检修日历核对。')

    return '\n'.join(parts)


def generate_railway_semantic(
    history: np.ndarray,
    pred: np.ndarray,
    col_names: List[str],
    pred_len: Optional[int] = None,
) -> str:
    """
    对外接口：根据历史与预测（均为原始量纲）生成铁路业务语义解释。
    history: (T_hist, C), pred: (T_pred, C)
    """
    return railway_semantic_from_fused(history, pred, col_names, pred_len=pred_len)


def generate_prototype_reasoning(
    reprogramming_attn: Union[np.ndarray, Any],
    prototype_words_by_id: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    sample_idx: int = 0,
    top_k: int = 5,
) -> str:
    """
    基于重编程层 patch→原型 的 attention 权重，生成「文本原型推理」段落，与事件推演结合可解释。

    reprogramming_attn: (B, H, L, S)，B=batch, H=heads, L=patch数, S=原型数
    prototype_words_by_id: 来自 prototype_topk_words.json，key 为原型 id，value 为 [{"word", "weight"}, ...]
                          若为 None，仅输出原型 ID 与权重，不解析词。
    sample_idx: 取第几个样本的 attention
    top_k: 取激活最高的前 k 个原型

    返回：一段「【重编程层文本原型】…」的中文描述。
    """
    if hasattr(reprogramming_attn, 'cpu'):
        reprogramming_attn = reprogramming_attn.cpu().numpy()
    arr = np.asarray(reprogramming_attn, dtype=np.float64)
    if arr.ndim != 4:
        return '【重编程层文本原型】attention 形状异常，无法生成原型推理。'
    # (B, H, L, S) -> 对 H、L 取平均，得到 (B, S)
    importance = np.mean(arr, axis=(1, 2))
    s_idx = min(sample_idx, importance.shape[0] - 1)
    proto_importance = importance[s_idx]
    top_indices = np.argsort(-proto_importance)[:top_k]

    parts = ['【重编程层文本原型】']
    parts.append('本窗口下激活最高的 {} 个文本原型（patch 对原型的平均注意力）如下，可与上方趋势/品种关系相互印证：'.format(top_k))
    for rank, pid in enumerate(top_indices, 1):
        w = float(proto_importance[pid])
        if prototype_words_by_id is not None and int(pid) in prototype_words_by_id:
            words = prototype_words_by_id[int(pid)]
            word_str = '、'.join([x.get('word', str(x)) for x in words[:5]])
            parts.append('  {}. 原型 {}（权重 {:.4f}）对应语义词：{}'.format(rank, int(pid), w, word_str))
        else:
            parts.append('  {}. 原型 {}（权重 {:.4f}）'.format(rank, int(pid), w))
    return '\n'.join(parts)
