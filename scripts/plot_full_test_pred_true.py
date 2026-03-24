#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot full test-period prediction vs. ground truth from run_eval outputs.

Prerequisite:
  Run evaluation with saving enabled:
    python run_eval.py ... --save_pred_true
  This will save pred.npy / true.npy under the checkpoint directory.

Usage:
  python scripts/plot_full_test_pred_true.py --ckpt_dir /path/to/ckpt_dir
  python scripts/plot_full_test_pred_true.py --ckpt_dir /path/to/ckpt_dir --mode step1
  python scripts/plot_full_test_pred_true.py --ckpt_dir /path/to/ckpt_dir --mode mean --n_plot 4

Modes:
  - step1: use only 1-step-ahead from each window => length = n_windows
  - mean:  stitch all horizons onto one timeline and average overlaps
  - last:  stitch all horizons and keep the latest available prediction for overlaps
"""

import argparse
import os
import numpy as np


def _choose_global_scale(y_true: np.ndarray, y_pred: np.ndarray, n_plot: int):
    """
    Use one shared magnitude for all subplots to keep y-units consistent.
    Returns (scale_factor, y_label_suffix).
    """
    y = np.concatenate([y_true[:, :n_plot].reshape(-1), y_pred[:, :n_plot].reshape(-1)], axis=0)
    y = np.asarray(y, dtype=float)
    max_abs = float(np.nanmax(np.abs(y))) if y.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0.0:
        return 1.0, ""
    exp = int(np.floor(np.log10(max_abs) / 3) * 3)
    exp = max(0, min(exp, 12))
    if exp == 0:
        return 1.0, ""
    if exp == 6:
        return 1e6, "(Millions)"
    if exp == 9:
        return 1e9, "(Billions)"
    return float(10 ** exp), rf"($\times 10^{{{exp}}}$)"


def _load_pred_true(ckpt_dir: str, tag: str):
    suffix = f"_{tag}" if tag else ""
    pred_path = os.path.join(ckpt_dir, f"pred{suffix}.npy")
    true_path = os.path.join(ckpt_dir, f"true{suffix}.npy")
    if not os.path.isfile(pred_path) or not os.path.isfile(true_path):
        raise FileNotFoundError(
            f"Missing {os.path.basename(pred_path)} / {os.path.basename(true_path)} under {ckpt_dir}. "
            f"Please re-run eval with --save_pred_true (and same --output_tag if used)."
        )
    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.shape != true.shape:
        raise ValueError(f"pred shape {pred.shape} != true shape {true.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected pred/true with shape (n_windows, pred_len, n_channels), got {pred.shape}")
    return pred, true


def _reconstruct_series(pred: np.ndarray, true: np.ndarray, mode: str):
    """
    pred/true: (W, L, C)
    Returns: (y_pred, y_true) with shape (T, C)
    """
    W, L, C = pred.shape
    if mode == "step1":
        return pred[:, 0, :], true[:, 0, :]

    T = W + L - 1
    y_true = np.full((T, C), np.nan, dtype=np.float64)
    if mode == "mean":
        acc = np.zeros((T, C), dtype=np.float64)
        cnt = np.zeros((T, C), dtype=np.float64)
        for w in range(W):
            for k in range(L):
                t = w + k
                acc[t] += pred[w, k]
                cnt[t] += 1.0
                if np.isnan(y_true[t]).any():
                    # fill per-channel if missing
                    miss = np.isnan(y_true[t])
                    y_true[t, miss] = true[w, k, miss]
        y_pred = acc / np.maximum(cnt, 1.0)
        return y_pred, y_true

    if mode == "last":
        y_pred = np.full((T, C), np.nan, dtype=np.float64)
        for w in range(W):
            for k in range(L):
                t = w + k
                y_pred[t] = pred[w, k]
                if np.isnan(y_true[t]).any():
                    miss = np.isnan(y_true[t])
                    y_true[t, miss] = true[w, k, miss]
        return y_pred, y_true

    raise ValueError(f"Unknown mode: {mode}")


def _infer_col_names(ckpt_dir: str, n_channels: int):
    # Try to reuse names from pred_true_4channels_data.npz if present.
    npz = os.path.join(ckpt_dir, "pred_true_4channels_data.npz")
    if os.path.isfile(npz):
        try:
            data = np.load(npz, allow_pickle=True)
            cn = data.get("col_names")
            if cn is not None:
                names = cn.tolist() if getattr(cn, "ndim", 0) >= 1 else [str(cn)]
                names = [str(x) for x in names]
                if len(names) >= n_channels:
                    return names[:n_channels]
                if len(names) > 0:
                    # pad
                    out = names[:]
                    while len(out) < n_channels:
                        out.append(f"Ch{len(out)}")
                    return out
        except Exception:
            pass
    return [f"Ch{i}" for i in range(n_channels)]


def _to_english_titles(names: list[str]) -> list[str]:
    # Common mapping for the Iron dataset.
    name_en = {
        "铁矿石": "Iron ore",
        "铁矿砂": "Iron sand",
        "铁矿粉": "Iron powder",
        "铁精矿粉": "Iron concentrate",
    }
    out = []
    for n in names:
        s = str(n).strip()
        out.append(name_en.get(s, s))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot full test prediction vs true from pred.npy/true.npy")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint dir that contains pred.npy / true.npy")
    parser.add_argument("--tag", type=str, default="", help="output_tag used in run_eval.py (if any)")
    parser.add_argument("--mode", type=str, default="mean", choices=["mean", "last", "step1"], help="reconstruction mode")
    parser.add_argument("--n_plot", type=int, default=4, help="plot first N channels")
    parser.add_argument("--out_name", type=str, default="", help="output filename stem (no extension)")
    parser.add_argument("--dpi", type=int, default=200, help="PNG dpi")
    parser.add_argument("--style", type=str, default="paper", choices=["paper", "default"], help="plot style")
    parser.add_argument(
        "--palette",
        type=str,
        default="classic",
        choices=["classic", "color", "print"],
        help="line colors: classic=blue/orange; color=colorblind-friendly; print=grayscale",
    )
    parser.add_argument("--x_label", type=str, default="Test timeline index", help="x-axis label")
    parser.add_argument("--y_label", type=str, default="Shipment", help="shared y-axis label (without scale suffix)")
    parser.add_argument(
        "--english",
        action="store_true",
        help="use English titles for known Chinese channel names (recommended for papers)",
    )
    parser.add_argument(
        "--no_append_tag",
        action="store_true",
        help="when --out_name is given, do not auto-append --tag to filename",
    )
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    pred, true = _load_pred_true(ckpt_dir, args.tag.strip())
    y_pred, y_true = _reconstruct_series(pred, true, args.mode)

    T, C = y_pred.shape
    n_plot = int(max(1, min(args.n_plot, C)))
    col_names = _infer_col_names(ckpt_dir, C)[:n_plot]
    if args.english:
        col_names = _to_english_titles(col_names)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.style == "paper":
        try:
            import scienceplots  # noqa: F401
            plt.style.use(["science", "ieee"])
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")
        # Match the paper style used by scripts/plot_pred_true_4channels_from_data.py
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.color": "#cfcfcf",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.30,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "svg.fonttype": "none",
        })

    # Layout
    ncols = 2 if n_plot <= 4 else 5
    nrows = int(np.ceil(n_plot / ncols))
    # Bigger than the 48-step plot: full-test curves are long and need more pixels.
    figsize = (10.5, 7.4) if n_plot <= 4 else (min(16.0, 2.8 * ncols), min(11.0, 2.8 * nrows + 1.0))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=260, sharex=True)
    axes = np.array(axes).reshape(-1)

    t = np.arange(T)
    # Colors
    if args.palette == "print":
        color_true, color_pred = "#111111", "#666666"
    elif args.palette == "color":
        # Okabe-Ito (colorblind-friendly)
        color_true, color_pred = "#0072B2", "#D55E00"
    else:
        # classic Matplotlib Tab10
        color_true, color_pred = "#1f77b4", "#ff7f0e"

    # Shared y-scale label to avoid unreadable "1e6" offset text on each subplot.
    scale_factor, y_suffix = _choose_global_scale(y_true, y_pred, n_plot=n_plot)

    # x ticks: keep readable for long series
    n_xticks = 6
    xticks = np.linspace(0, max(0, T - 1), num=n_xticks, dtype=int)

    for c in range(n_plot):
        ax = axes[c]
        ax.plot(
            t,
            y_true[:, c] / scale_factor,
            color=color_true,
            linewidth=1.9,
            label="True",
            alpha=0.95,
        )
        ax.plot(
            t,
            y_pred[:, c] / scale_factor,
            color=color_pred,
            linewidth=2.2,
            linestyle="--",
            label="Pred",
            alpha=0.92,
        )
        # Leave fontsize to rcParams (axes.titlesize) for consistency with other plots.
        ax.set_title(str(col_names[c]), fontweight="medium")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.set_xlim(-0.5, T - 0.5)
        ax.get_yaxis().get_offset_text().set_visible(False)
        ax.set_xticks(xticks)
        # Leave tick label sizes to rcParams for consistency.
        ax.tick_params(axis="both")
        if c >= (nrows - 1) * ncols:
            ax.set_xlabel(args.x_label)

    for k in range(n_plot, nrows * ncols):
        axes[k].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))

    ylab = args.y_label.strip()
    if y_suffix:
        ylab = f"{ylab} {y_suffix}"
    fig.supylabel(ylab, x=0.02)

    title_mode = {"mean": "mean-overlap", "last": "last-overlap", "step1": "step1-only"}[args.mode]
    out_stem = args.out_name.strip() or f"pred_true_full_{title_mode}"
    # Avoid duplicated suffix like "..._noZT_full_noZT_full" when user already encodes tag in out_name.
    if args.tag.strip() and not (args.out_name.strip() and args.no_append_tag):
        out_stem += f"_{args.tag.strip()}"
    out_png = os.path.join(ckpt_dir, f"{out_stem}.png")
    out_svg = os.path.join(ckpt_dir, f"{out_stem}.svg")
    out_pdf = os.path.join(ckpt_dir, f"{out_stem}.pdf")
    fig.subplots_adjust(top=0.90, left=0.08, right=0.99, bottom=0.08, hspace=0.22, wspace=0.25)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out_png, ",", out_svg, ",", out_pdf)


if __name__ == "__main__":
    main()

