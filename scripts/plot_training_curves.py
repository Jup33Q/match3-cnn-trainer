#!/usr/bin/env python3
"""plot_training_curves.py - 绘制美观的训练曲线（处理跨 stage、跨数量级差异）"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# --- 配色方案 ---
COLORS = {
    "total": "#1f77b4",    # 蓝
    "dice": "#ff7f0e",     # 橙
    "focal": "#2ca02c",    # 绿
    "boundary": "#d62728", # 红
    "iou": "#9467bd",      # 紫
    "f1": "#8c564b",       # 棕
    "precision": "#e377c2", # 粉
    "recall": "#7f7f7f",    # 灰
    "lr": "#bcbd22",        # 黄绿
}

STAGE_BG_COLORS = {
    "stage0": "#e8f4f8",
    "stage1": "#fff4e6",
    "stage2": "#e8f5e9",
    "stage3": "#fce4ec",
    "stage4": "#f3e5f5",
    "stage5": "#e8eaf6",
}

STAGE_EDGE_COLORS = {
    "stage0": "#4fc3f7",
    "stage1": "#ffb74d",
    "stage2": "#81c784",
    "stage3": "#f06292",
    "stage4": "#ba68c8",
    "stage5": "#7986cb",
}


def load_data(json_path: str = "logs/training_loss.json") -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, path: str, dpi: int = 200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def _extract_xy(series: List[Dict]) -> (np.ndarray, np.ndarray):
    """提取 x(step/epoch) 和 y(value)，过滤非正数值（log 坐标安全）"""
    xs = np.array([s["step"] for s in series])
    ys = np.array([s["value"] for s in series])
    # 对 log scale，需要所有值 > 0；用极小值替换 0/负数
    ys = np.where(ys <= 0, np.nan, ys)
    return xs, ys


def _smooth(y: np.ndarray, window: int = 50) -> np.ndarray:
    """简单移动平均平滑"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(np.nan_to_num(y, nan=np.nanmedian(y)), kernel, mode="same")
    return smoothed


def plot_global_step_overview(data: Dict, out_dir: str):
    """图1: 全局 Step 级 Total Loss 概览（log y，stage 背景色区分）"""
    step_data = data.get("step_level", {})
    if "total" not in step_data or not step_data["total"]:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # 画 stage 背景色带
    stage_info = data.get("stage_level", {})
    stages = sorted(stage_info.keys(), key=lambda s: int(s.replace("stage", "")))

    # 从 step 数据推断 stage 边界
    all_steps = [s["step"] for s in step_data["total"]]
    min_step, max_step = min(all_steps), max(all_steps)

    for stage in stages:
        train_total = stage_info[stage].get("train", {}).get("total", [])
        if not train_total:
            continue
        s_min = min(s["step"] for s in train_total)
        s_max = max(s["step"] for s in train_total)
        bg = STAGE_BG_COLORS.get(stage, "#f5f5f5")
        edge = STAGE_EDGE_COLORS.get(stage, "#999")
        ax.axvspan(s_min - 0.5, s_max + 0.5, alpha=0.25, color=bg, zorder=0)
        # stage 标签
        ax.text((s_min + s_max) / 2, 0.98, stage.upper(),
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, fontweight="bold",
                color=edge, alpha=0.8)

    # 画原始数据（半透明）
    xs, ys = _extract_xy(step_data["total"])
    valid_mask = ~np.isnan(ys)
    ax.plot(xs[valid_mask], ys[valid_mask], color=COLORS["total"], alpha=0.15, lw=0.8, zorder=1)

    # 画平滑曲线
    ys_smooth = _smooth(ys, window=min(100, max(20, len(ys) // 30)))
    ax.plot(xs[valid_mask], ys_smooth[valid_mask], color=COLORS["total"], lw=2.2, zorder=2, label="Total Loss (smoothed)")

    ax.set_yscale("log")
    ax.set_ylabel("Total Loss (log scale)", fontsize=12)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_title("Global Training Loss Overview (Step-level)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlim(min_step, max_step)

    # y 轴科学计数法
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    save_fig(fig, os.path.join(out_dir, "01_global_step_overview.png"))


def plot_stage_train_loss(data: Dict, out_dir: str):
    """图2: 每个 Stage 的训练损失详情（log y，多指标同图）"""
    stage_info = data.get("stage_level", {})
    stages = sorted(stage_info.keys(), key=lambda s: int(s.replace("stage", "")))

    for stage in stages:
        train = stage_info[stage].get("train", {})
        metrics = [k for k in ["total", "dice", "focal", "boundary"] if k in train and train[k]]
        if not metrics:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")

        for metric in metrics:
            xs, ys = _extract_xy(train[metric])
            valid_mask = ~np.isnan(ys)
            if not valid_mask.any():
                continue
            ax.plot(xs[valid_mask], ys[valid_mask], marker="o", markersize=4,
                    color=COLORS[metric], lw=1.8, label=metric.upper(), zorder=2)

        ax.set_yscale("log")
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_title(f"{stage.upper()} — Training Loss Breakdown", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

        save_fig(fig, os.path.join(out_dir, f"02_{stage}_train_loss.png"))


def plot_stage_val_metrics(data: Dict, out_dir: str):
    """图3: 验证指标（线性 y，0~1 范围，多 stage 对比）"""
    stage_info = data.get("stage_level", {})
    stages = sorted(stage_info.keys(), key=lambda s: int(s.replace("stage", "")))

    metrics_to_plot = ["iou", "f1", "precision", "recall"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        for stage in stages:
            val = stage_info[stage].get("val", {})
            if metric not in val or not val[metric]:
                continue
            xs, ys = _extract_xy(val[metric])
            valid_mask = ~np.isnan(ys)
            ax.plot(xs[valid_mask], ys[valid_mask], marker="s", markersize=5,
                    color=STAGE_EDGE_COLORS.get(stage, "#333"), lw=2,
                    label=stage.upper(), zorder=2)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(f"Validation {metric.upper()}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle("Validation Metrics Across Stages", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "03_validation_metrics.png"))


def plot_step_level_detail(data: Dict, out_dir: str):
    """图4: Step 级损失分面图（每个 metric 一个 subplot，log y）"""
    step_data = data.get("step_level", {})
    metrics = [k for k in ["total", "dice", "focal", "boundary"] if k in step_data and step_data[k]]
    if not metrics:
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, metrics):
        series = step_data[metric]
        xs, ys = _extract_xy(series)
        valid_mask = ~np.isnan(ys)

        # 原始数据（淡色）
        ax.plot(xs[valid_mask], ys[valid_mask], color=COLORS[metric], alpha=0.15, lw=0.6)
        # 平滑曲线
        ys_smooth = _smooth(ys, window=min(100, max(20, len(ys) // 30)))
        ax.plot(xs[valid_mask], ys_smooth[valid_mask], color=COLORS[metric], lw=2, label=metric.upper())

        ax.set_yscale("log")
        ax.set_ylabel(f"{metric.upper()} Loss", fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    axes[-1].set_xlabel("Step", fontsize=12)
    fig.suptitle("Step-level Loss Components (Log Scale)", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "04_step_level_detail.png"))


def plot_lr_curve(data: Dict, out_dir: str):
    """图5: 学习率变化曲线"""
    step_data = data.get("step_level", {})
    if "lr" not in step_data or not step_data["lr"]:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")

    xs, ys = _extract_xy(step_data["lr"])
    valid_mask = ~np.isnan(ys)
    ax.plot(xs[valid_mask], ys[valid_mask], color=COLORS["lr"], lw=2)

    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    save_fig(fig, os.path.join(out_dir, "05_learning_rate.png"))


def plot_combined_stage_loss_with_inset(data: Dict, out_dir: str):
    """图6: 所有 stage 的 total loss 合并图 + 放大 inset（处理数量级跳跃）"""
    stage_info = data.get("stage_level", {})
    stages = sorted(stage_info.keys(), key=lambda s: int(s.replace("stage", "")))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    all_xs, all_ys = [], []
    for stage in stages:
        train = stage_info[stage].get("train", {})
        if "total" not in train or not train["total"]:
            continue
        xs, ys = _extract_xy(train["total"])
        valid_mask = ~np.isnan(ys)
        ax.plot(xs[valid_mask], ys[valid_mask], marker="o", markersize=5,
                color=STAGE_EDGE_COLORS.get(stage, "#333"), lw=2.2,
                label=stage.upper(), zorder=2)
        all_xs.extend(xs[valid_mask].tolist())
        all_ys.extend(ys[valid_mask].tolist())

    ax.set_yscale("log")
    ax.set_ylabel("Total Loss (log scale)", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title("Total Loss by Stage (Epoch-level, Log Scale)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    # 添加 inset：放大低 loss 区域（如果有足够的数据点）
    if len(all_ys) > 5:
        # 找到中位数以下的区域作为 inset
        sorted_ys = sorted([y for y in all_ys if not np.isnan(y)])
        if len(sorted_ys) > 10:
            low_threshold = sorted_ys[int(len(sorted_ys) * 0.3)]  # 下 30% 分位
            high_threshold = sorted_ys[int(len(sorted_ys) * 0.7)]  # 上 70% 分位
            if low_threshold != high_threshold:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                axins = inset_axes(ax, width="40%", height="35%", loc="upper right",
                                   bbox_to_anchor=(0.55, 0.55, 0.43, 0.43), bbox_transform=ax.transAxes)
                for stage in stages:
                    train = stage_info[stage].get("train", {})
                    if "total" not in train or not train["total"]:
                        continue
                    xs, ys = _extract_xy(train["total"])
                    valid_mask = ~np.isnan(ys)
                    axins.plot(xs[valid_mask], ys[valid_mask], marker="o", markersize=3,
                               color=STAGE_EDGE_COLORS.get(stage, "#333"), lw=1.5)
                axins.set_ylim(low_threshold * 0.5, high_threshold * 2)
                axins.set_yscale("log")
                axins.grid(True, ls="--", alpha=0.2)
                axins.set_title("Zoom: low-loss region", fontsize=9)
                axins.tick_params(labelsize=8)
                axins.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())

    save_fig(fig, os.path.join(out_dir, "06_combined_epoch_loss.png"))


def main():
    data = load_data()
    out_dir = "logs/training_curves"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[PLOT] 输出目录: {out_dir}")

    plot_global_step_overview(data, out_dir)
    plot_stage_train_loss(data, out_dir)
    plot_stage_val_metrics(data, out_dir)
    plot_step_level_detail(data, out_dir)
    plot_lr_curve(data, out_dir)
    plot_combined_stage_loss_with_inset(data, out_dir)

    print("[PLOT] 全部完成！")


if __name__ == "__main__":
    main()
