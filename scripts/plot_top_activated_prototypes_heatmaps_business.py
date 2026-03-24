#!/usr/bin/env python3
"""
Plot heatmaps for the *top-activated* text prototypes (by reprogramming attention)
under business-oriented word sets.

Motivation:
- plot_prototype_word_heatmaps_business.py shows prototype 0-9 by index (not "importance").
- Here we compute importance on a real test window:
    reprogramming_attn: (B, H, L, S) -> mean over H and L -> (S,)
  then take top-k prototypes and visualize their W rows against chosen word sets.

Outputs (saved under ckpt_dir/prototype_export by default):
- top_activated_prototypes_word_sets_heatmaps_business.png/svg
- top_activated_prototypes_word_sets_heatmaps_business_cn.png/svg

Usage:
  python scripts/plot_top_activated_prototypes_heatmaps_business.py --ckpt_dir checkpoints/...-iron_stage2_linear
  python scripts/plot_top_activated_prototypes_heatmaps_business.py --ckpt_dir checkpoints/...-iron_stage2_randomllm_linear
"""

import argparse
import os
import sys
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# Same word sets as plot_prototype_word_heatmaps_business.py (supports multi-token aggregation).
BUSINESS_WORD_SETS: List[Tuple[str, List[Tuple[Union[int, List[int]], str]]]] = [
    (
        'Temporal & periodic dynamics',
        [
            (931, 'time'),
            (3785, 'period'),
            (14218, 'daily'),
            (11015, 'minute'),
            (20410, 'schedule'),
            (4259, 'season'),
            (9055, 'delay'),
        ],
    ),
    (
        'Volume & extreme variations',
        [
            (7977, 'volume'),
            (5225, 'zero'),
            (4069, 'empty'),
            (19224, 'peak'),
            (2254, 'load'),
            (13284, 'capacity'),
            ([3273, 482], 'shortage'),
        ],
    ),
    (
        'Railway & mining entities',
        [
            (7903, 'mine'),
            (7945, 'train'),
            ([3005, 523], 'freight'),
            (5073, 'station'),
            (5782, 'route'),
            (13916, 'dispatch'),
            (17040, 'cargo'),
        ],
    ),
]


def get_parser():
    p = argparse.ArgumentParser(description="Plot top-activated prototypes heatmaps (business word sets)")
    p.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint directory containing a 'checkpoint' file")
    p.add_argument("--top_k", type=int, default=10, help="number of top activated prototypes to plot")
    p.add_argument("--sample_idx", type=int, default=0, help="which sample in the first eval batch to use")
    p.add_argument("--device", type=str, default="", help="cuda/cpu/empty(auto)")
    p.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size for grabbing attention")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--cn", action="store_true", help="also output Chinese-title version")
    p.add_argument("--title_suffix", type=str, default="", help="extra title suffix")
    return p


def infer_model_comment(ckpt_dir: str) -> str:
    base = os.path.basename(ckpt_dir.rstrip("/"))
    # training saves: {setting}-{model_comment}
    if "-" in base:
        return base.split("-")[-1]
    return base


def load_W(ckpt_dir: str) -> np.ndarray:
    export_dir = os.path.join(ckpt_dir, "prototype_export")
    W_path = os.path.join(export_dir, "prototype_word_weight_matrix.npy")
    if not os.path.isfile(W_path):
        raise FileNotFoundError(f"Missing {W_path}. Run extract_reprogramming_prototypes.py first.")
    return np.load(W_path), export_dir


def aggregate_W_cols(W_rows: np.ndarray, word_list: List[Tuple[Union[int, List[int]], str]]) -> np.ndarray:
    """W_rows: (n_proto, vocab_size) -> returns (n_proto, n_words) aggregated columns."""
    cols = []
    for v, _ in word_list:
        if isinstance(v, (list, tuple, np.ndarray)):
            ids = [int(x) for x in v]
            cols.append(np.mean(W_rows[:, np.array(ids)], axis=1, keepdims=True))
        else:
            cols.append(W_rows[:, np.array([int(v)])])
    return np.concatenate(cols, axis=1) if cols else W_rows[:, :0]


def compute_top_activated_prototypes(ckpt_dir: str, device: torch.device, eval_batch_size: int, sample_idx: int, top_k: int) -> np.ndarray:
    from models import TimeLLM
    from utils.tools import load_content
    from data_provider.data_factory import data_provider

    ckpt_path = os.path.join(ckpt_dir, "checkpoint")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")

    # Minimal args consistent with Iron experiments (same as extract_reprogramming_prototypes.py)
    class Args:
        task_name = "long_term_forecast"
        model_id = "Iron_96_48"
        model = "TimeLLM"
        data = "custom"
        root_path = "./dataset/"
        data_path = "2023_2025_Iron_data.csv"
        features = "M"
        seq_len = 96
        label_len = 48
        pred_len = 48
        enc_in = 4
        dec_in = 4
        c_out = 4
        d_model = 32
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 128
        factor = 3
        embed = "timeF"
        des = "Iron_Ore_Transport_Exp"
        patch_len = 16
        stride = 8
        dropout = 0.1
        llm_model = "LLAMA"
        llm_dim = 4096
        llm_layers = 32
        prompt_type = "full"
        prompt_domain = 0
        model_comment = infer_model_comment(ckpt_dir)

        # must match for multivariate checkpoints (we keep False here; user can extend if needed)
        multivariate = False
        channel_mixing = False

        # keep reprogramming (need attention)
        ablate_reprogramming = False
        ablate_prompt = False
        ablate_prompt_description = False
        ablate_prompt_task = False
        ablate_prompt_stats = False

        # aux not needed for attention
        use_aux_loss = False

    args = Args()
    args.content = load_content(args)

    model = TimeLLM.Model(args).float()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    test_set, _ = data_provider(args, "test")
    loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(loader))
    if len(batch) == 5:
        batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
    else:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)

    with torch.no_grad():
        out, extra = model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark, return_reprogramming_attention=True, return_aux_repr=False
        )
        attn = extra.get("reprogramming_attn")
        if attn is None:
            raise RuntimeError("No reprogramming_attn returned. Is ablate_reprogramming enabled?")
        if hasattr(attn, "cpu"):
            attn = attn.cpu().numpy()
        attn = np.asarray(attn, dtype=np.float64)  # (B, H, L, S)
        if attn.ndim != 4:
            raise RuntimeError(f"Unexpected attention shape: {attn.shape}")
        s_idx = min(max(sample_idx, 0), attn.shape[0] - 1)
        importance = attn[s_idx].mean(axis=(0, 1))  # (S,)
        top_ids = np.argsort(-importance)[:top_k]
        return top_ids.astype(int)


def plot_and_save(W: np.ndarray, export_dir: str, proto_ids: np.ndarray, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    W_rows = W[np.array(proto_ids)]
    titles_en = [t for t, _ in BUSINESS_WORD_SETS]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (_, word_list), title in zip(axes, BUSINESS_WORD_SETS, titles_en):
        labels = [lbl for _, lbl in word_list]
        W_sub = aggregate_W_cols(W_rows, word_list)
        sns.heatmap(
            W_sub,
            xticklabels=labels,
            yticklabels=[str(int(i)) for i in proto_ids],
            ax=ax,
            cmap="RdYlBu_r",
            center=0,
            cbar_kws={"label": "weight"},
            annot=False,
        )
        ax.set_xlabel("Word")
        ax.set_ylabel("Prototype (top-activated)")
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if args.title_suffix:
        fig.suptitle(args.title_suffix, fontsize=9, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(export_dir, "top_activated_prototypes_word_sets_heatmaps_business.png")
    out_svg = os.path.join(export_dir, "top_activated_prototypes_word_sets_heatmaps_business.svg")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out_png)

    if args.cn:
        titles_cn = ["时间与周期规律", "量级与极端波动", "重载铁路物理实体"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (_, word_list), title in zip(axes, BUSINESS_WORD_SETS, titles_cn):
            labels = [lbl for _, lbl in word_list]
            W_sub = aggregate_W_cols(W_rows, word_list)
            sns.heatmap(
                W_sub,
                xticklabels=labels,
                yticklabels=[str(int(i)) for i in proto_ids],
                ax=ax,
                cmap="RdYlBu_r",
                center=0,
                cbar_kws={"label": "weight"},
                annot=False,
            )
            ax.set_xlabel("Word")
            ax.set_ylabel("Prototype (top-activated)")
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        if args.title_suffix:
            fig.suptitle(args.title_suffix, fontsize=9, y=1.02)
        plt.tight_layout()
        out_cn_png = os.path.join(export_dir, "top_activated_prototypes_word_sets_heatmaps_business_cn.png")
        out_cn_svg = os.path.join(export_dir, "top_activated_prototypes_word_sets_heatmaps_business_cn.svg")
        fig.savefig(out_cn_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(out_cn_svg, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("saved:", out_cn_png)


def main():
    args = get_parser().parse_args()
    ckpt_dir = args.ckpt_dir.rstrip("/")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W, export_dir = load_W(ckpt_dir)
    proto_ids = compute_top_activated_prototypes(
        ckpt_dir, device=device, eval_batch_size=args.eval_batch_size, sample_idx=args.sample_idx, top_k=args.top_k
    )
    title = args.title_suffix or os.path.basename(ckpt_dir)
    args.title_suffix = f"{title}  [top{args.top_k} by reprogramming attn]"
    plot_and_save(W, export_dir, proto_ids, args)


if __name__ == "__main__":
    main()

