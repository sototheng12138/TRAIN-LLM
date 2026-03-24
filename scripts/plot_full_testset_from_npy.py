import argparse
import os
from typing import List, Tuple

import numpy as np


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_col_names(n: int) -> List[str]:
    if n == 4:
        return ["铁矿石", "铁矿砂", "铁矿粉", "铁精矿粉"]
    return [f"指标{i+1}" for i in range(n)]

def _default_col_names_en(n: int) -> List[str]:
    if n == 4:
        return ["Iron ore", "Iron sand", "Iron powder", "Iron concentrate"]
    return [f"Series {i+1}" for i in range(n)]


def _flatten_pred_true(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accept shapes like:
      - (n_windows, pred_len, n_channels)
      - (n_total, n_channels)
    Return flattened (n_points, n_channels).
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: pred={pred.shape}, true={true.shape}")
    if pred.ndim == 3:
        n_win, pred_len, n_ch = pred.shape
        return pred.reshape(n_win * pred_len, n_ch), true.reshape(n_win * pred_len, n_ch)
    if pred.ndim == 2:
        return pred, true
    raise ValueError(f"Unsupported pred ndim={pred.ndim}, shape={pred.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint directory containing pred*.npy / true*.npy")
    ap.add_argument("--tag", type=str, default="noZT_full", help="suffix tag used in filenames, e.g. noZT_full")
    ap.add_argument("--max_points", type=int, default=20000, help="max points per channel for scatter (subsample if larger)")
    ap.add_argument("--style", type=str, default="default", choices=["default", "paper_v4"], help="plot style preset")
    args = ap.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    tag = (args.tag or "").strip()
    tag_suffix = f"_{tag}" if tag else ""

    pred_path = os.path.join(ckpt_dir, f"pred{tag_suffix}.npy")
    true_path = os.path.join(ckpt_dir, f"true{tag_suffix}.npy")
    # Baselines may save as pred_full.npy/true_full.npy for convenience.
    if not (os.path.exists(pred_path) and os.path.exists(true_path)) and tag == "full":
        pred_path = os.path.join(ckpt_dir, "pred_full.npy")
        true_path = os.path.join(ckpt_dir, "true_full.npy")
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        raise FileNotFoundError(f"Missing npy files: {pred_path} / {true_path}")

    pred = np.load(pred_path, allow_pickle=False)
    true = np.load(true_path, allow_pickle=False)
    pred2d, true2d = _flatten_pred_true(pred, true)

    n_points, n_ch = pred2d.shape
    col_names = _default_col_names(n_ch)
    col_names_en = _default_col_names_en(n_ch)
    if pred.ndim != 3:
        raise ValueError(
            f"Expected saved pred/true as (n_windows, pred_len, n_channels) for time-aligned plots; got pred shape={pred.shape}"
        )
    n_windows, pred_len, _ = pred.shape

    # Metrics over all points (original scale in these npy)
    err = pred2d - true2d
    mae_per = np.mean(np.abs(err), axis=0)
    rmse_per = np.sqrt(np.mean(err ** 2, axis=0))
    mae_all = float(np.mean(np.abs(err)))
    rmse_all = float(np.sqrt(np.mean(err ** 2)))

    out_dir = os.path.join(ckpt_dir, "plots_full_testset")
    _safe_makedirs(out_dir)

    # Lazy import matplotlib (so loading npy works even if mpl missing)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.style == "paper_v4":
        # Match the requested v4 look: clean grid, classic colors, serif titles.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["legend.fontsize"] = 11
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["grid.alpha"] = 0.25
        plt.rcParams["grid.linestyle"] = "-"
    else:
        plt.rcParams["font.sans-serif"] = [
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "SimHei",
            "Microsoft YaHei",
            "SimSun",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

    # 1) Full flattened series (downsample for readability)
    stride = max(1, n_points // 8000)
    idx = np.arange(0, n_points, stride)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    axes = axes.flatten()
    for c in range(min(4, n_ch)):
        ax = axes[c]
        ax.plot(idx, true2d[idx, c], color="#1f77b4", linewidth=1.2, alpha=0.9, label="True")
        ax.plot(idx, pred2d[idx, c], color="#d62728", linewidth=1.0, alpha=0.8, linestyle="--", label="Pred")
        ax.set_title(f"{col_names[c]} | MAE={mae_per[c]:.2f}, RMSE={rmse_per[c]:.2f}")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Flattened test index (downsampled)")
        ax.set_ylabel("Value")
        ax.legend(fontsize=9, framealpha=0.9)
    for k in range(min(4, n_ch), 4):
        axes[k].axis("off")
    fig.suptitle(f"Full test set (flattened) | MAE={mae_all:.2f}, RMSE={rmse_all:.2f} | points={n_points}", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"full_series{tag_suffix}.png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, f"full_series{tag_suffix}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 1b) Step-1 series across windows (x-axis = n_windows, often what you expect as "172")
    step = 0
    xw = np.arange(n_windows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120, sharex=True)
    axes = axes.flatten()
    for c in range(min(4, n_ch)):
        ax = axes[c]
        if args.style == "paper_v4":
            ax.plot(xw, true[:, step, c], color="#1f77b4", linewidth=2.0, label="True")
            ax.plot(xw, pred[:, step, c], color="#ff7f0e", linewidth=2.0, linestyle="--", label="Pred")
            ax.set_title(col_names_en[c])
            ax.set_xlim(0, n_windows - 1)
            ax.set_xticks([0, 34, 68, 102, 136, n_windows - 1])
        else:
            ax.plot(xw, true[:, step, c], color="#1f77b4", linewidth=1.6, alpha=0.9, label="True (t+1)")
            ax.plot(xw, pred[:, step, c], color="#d62728", linewidth=1.2, alpha=0.85, linestyle="--", label="Pred (t+1)")
            ax.set_title(f"{col_names[c]} | step=t+1 | windows={n_windows}")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Test window index (≈ time)")
            ax.set_ylabel("Value")
            ax.legend(fontsize=9, framealpha=0.9)
    for k in range(min(4, n_ch), 4):
        axes[k].axis("off")
    if args.style == "paper_v4":
        # Single legend centered on top; shared labels.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
        fig.supylabel("Shipment (Millions)")
        fig.tight_layout(rect=[0.04, 0.04, 0.96, 0.93])
        fig.savefig(os.path.join(out_dir, f"pred_true_full_TEST_step1{tag_suffix}_paper_v4.png"), bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, f"pred_true_full_TEST_step1{tag_suffix}_paper_v4.svg"), bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, f"pred_true_full_TEST_step1{tag_suffix}_paper_v4.pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    else:
        fig.suptitle(f"Full test set (t+1 over windows) | n_windows={n_windows}, pred_len={pred_len}", y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"step1_series_over_windows{tag_suffix}.png"), bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, f"step1_series_over_windows{tag_suffix}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2) True vs Pred scatter (subsample if too many)
    rng = np.random.default_rng(0)
    if n_points > args.max_points:
        take = rng.choice(n_points, size=args.max_points, replace=False)
        take.sort()
    else:
        take = np.arange(n_points)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    axes = axes.flatten()
    for c in range(min(4, n_ch)):
        ax = axes[c]
        x = true2d[take, c]
        y = pred2d[take, c]
        ax.scatter(x, y, s=4, alpha=0.25, color="#4c78a8", edgecolors="none")
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], color="#d62728", linewidth=1.2, alpha=0.9)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(f"{col_names[c]} | n={len(take)}")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    for k in range(min(4, n_ch), 4):
        axes[k].axis("off")
    fig.suptitle(f"True vs Pred (subsampled) | tag={tag or 'none'}", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"scatter_true_vs_pred{tag_suffix}.png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, f"scatter_true_vs_pred{tag_suffix}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3) Error distribution per channel
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    axes = axes.flatten()
    for c in range(min(4, n_ch)):
        ax = axes[c]
        e = err[:, c]
        # robust binning around percentiles
        p1, p99 = np.percentile(e, [1, 99])
        lo, hi = float(p1), float(p99)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = float(np.min(e)), float(np.max(e))
        bins = 60
        ax.hist(np.clip(e, lo, hi), bins=bins, color="#72b7b2", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="#333333", linewidth=1.0, alpha=0.9)
        ax.set_title(f"{col_names[c]} | MAE={mae_per[c]:.2f}, RMSE={rmse_per[c]:.2f}")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Pred - True (clipped to 1-99% range)")
        ax.set_ylabel("Count")
    for k in range(min(4, n_ch), 4):
        axes[k].axis("off")
    fig.suptitle("Error distribution on full test set", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"error_hist{tag_suffix}.png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, f"error_hist{tag_suffix}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save a small metrics text for paper bookkeeping
    metrics_path = os.path.join(out_dir, f"metrics_full_testset{tag_suffix}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"points={n_points}, channels={n_ch}\n")
        f.write(f"MAE_all={mae_all:.6f}, RMSE_all={rmse_all:.6f}\n")
        for c in range(n_ch):
            f.write(f"{col_names[c]}: MAE={mae_per[c]:.6f}, RMSE={rmse_per[c]:.6f}\n")

    print("Saved plots to:", out_dir)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()

