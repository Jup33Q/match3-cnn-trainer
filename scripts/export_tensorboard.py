#!/usr/bin/env python3
"""export_tensorboard.py - 提取 TensorBoard event 文件中的训练损失，输出 JSON + Markdown 报告"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

from tensorboard.backend.event_processing import event_accumulator


def find_main_event_file(log_dir: str = "logs") -> str:
    """找最大的 event 文件（通常是当前活跃的那个）"""
    files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    files.sort(key=lambda x: -os.path.getsize(os.path.join(log_dir, x)))
    return os.path.join(log_dir, files[0])


def extract_scalars(ea: event_accumulator.EventAccumulator) -> Dict[str, List[Dict[str, Any]]]:
    """提取所有 scalar 数据，按 tag 分组"""
    result = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        result[tag] = [{"step": e.step, "value": float(e.value), "wall_time": e.wall_time} for e in events]
    return result


def build_json_report(data: Dict[str, List[Dict]], event_file: str) -> Dict[str, Any]:
    """构建结构化的 JSON 报告"""
    report = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "event_file": event_file,
            "event_file_size_bytes": os.path.getsize(event_file),
        },
        "step_level": {},
        "stage_level": {},
        "memory": {},
    }

    # Step-level (per-batch / per-step) 训练损失
    step_prefix = "train_step/"
    for tag, values in data.items():
        if tag.startswith(step_prefix):
            key = tag.replace(step_prefix, "")
            report["step_level"][key] = values
        elif tag.startswith("train_memory/"):
            key = tag.replace("train_memory/", "")
            report["memory"][key] = values

    # Stage-level (per-epoch) 数据
    for tag, values in data.items():
        if "/" not in tag or tag.startswith("train_"):
            continue
        parts = tag.split("/")
        if len(parts) != 2:
            continue
        stage, metric = parts
        if not stage.startswith("stage"):
            continue
        if stage not in report["stage_level"]:
            report["stage_level"][stage] = {"train": {}, "val": {}}
        category = "train" if metric.startswith("train_") else "val"
        short_metric = metric.replace("train_", "").replace("val_", "")
        report["stage_level"][stage][category][short_metric] = values

    # 自动识别有哪些 stage
    report["metadata"]["stages"] = sorted(report["stage_level"].keys())
    return report


def _fmt_sci(v: float) -> str:
    """科学计数法，保留 4 位有效数字"""
    return f"{v:.4e}"


def _fmt_float(v: float) -> str:
    """普通浮点，保留 4 位小数"""
    return f"{v:.4f}"


def build_md_report(report: Dict[str, Any]) -> str:
    """生成 Markdown 概览报告"""
    meta = report["metadata"]
    lines = []

    lines.append("# 🎮 Match-3 CNN Trainer — 训练日志概览")
    lines.append("")
    lines.append(f"- **导出时间**: {meta['exported_at']}")
    lines.append(f"- **数据来源**: `{meta['event_file']}` ({meta['event_file_size_bytes']:,} bytes)")
    lines.append(f"- **课程学习阶段**: {', '.join(meta.get('stages', []))}")
    lines.append("")

    # Stage-level 汇总
    lines.append("## 📊 Stage 级别训练指标（每 Epoch）")
    lines.append("")

    for stage in sorted(report["stage_level"].keys(), key=lambda x: int(x.replace("stage", ""))):
        sdata = report["stage_level"][stage]
        lines.append(f"### {stage.upper()}")
        lines.append("")

        # Train metrics table
        train = sdata.get("train", {})
        if train:
            # 找所有 metric 的共同长度
            metric_names = sorted(train.keys())
            n_epochs = max(len(train[m]) for m in metric_names) if metric_names else 0
            lines.append("#### 训练损失")
            lines.append("")
            lines.append("| Epoch | " + " | ".join(metric_names) + " |")
            lines.append("|------|" + "|".join(["------"] * len(metric_names)) + "|")
            for i in range(n_epochs):
                row_vals = []
                for m in metric_names:
                    vals = train[m]
                    if i < len(vals):
                        v = vals[i]["value"]
                        row_vals.append(_fmt_sci(v))
                    else:
                        row_vals.append("—")
                # epoch number from step (usually step=epoch number)
                epoch_num = train[metric_names[0]][i]["step"] if metric_names and i < len(train[metric_names[0]]) else i + 1
                lines.append(f"| {epoch_num} | " + " | ".join(row_vals) + " |")
            lines.append("")

        # Val metrics table
        val = sdata.get("val", {})
        if val:
            metric_names = sorted(val.keys())
            n_epochs = max(len(val[m]) for m in metric_names) if metric_names else 0
            lines.append("#### 验证指标")
            lines.append("")
            lines.append("| Epoch | " + " | ".join(metric_names) + " |")
            lines.append("|------|" + "|".join(["------"] * len(metric_names)) + "|")
            for i in range(n_epochs):
                row_vals = []
                for m in metric_names:
                    vals = val[m]
                    if i < len(vals):
                        v = vals[i]["value"]
                        row_vals.append(_fmt_float(v))
                    else:
                        row_vals.append("—")
                epoch_num = val[metric_names[0]][i]["step"] if metric_names and i < len(val[metric_names[0]]) else i + 1
                lines.append(f"| {epoch_num} | " + " | ".join(row_vals) + " |")
            lines.append("")

    # Step-level summary
    step = report.get("step_level", {})
    if step:
        lines.append("## 🔬 Step 级别训练损失（采样）")
        lines.append("")
        lines.append("以下展示每个 step 的原始损失值（科学计数法），便于观察训练动态：")
        lines.append("")
        for key in ["total", "dice", "focal", "boundary"]:
            if key not in step:
                continue
            vals = step[key]
            lines.append(f"### {key.upper()} Loss")
            lines.append("")
            # 只展示每 500 step 一个采样点，避免表格过长
            sample = vals[::max(1, len(vals) // 20)]
            lines.append("| Step | Value |")
            lines.append("|------|-------|")
            for item in sample:
                lines.append(f"| {item['step']} | {_fmt_sci(item['value'])} |")
            lines.append("")

    # Memory summary
    mem = report.get("memory", {})
    if mem:
        lines.append("## 💾 显存/内存监控")
        lines.append("")
        for key, vals in mem.items():
            if not vals:
                continue
            latest = vals[-1]["value"]
            peak = max(v["value"] for v in vals)
            lines.append(f"- **{key}**: 最新={latest:.1f}, 峰值={peak:.1f}")
        lines.append("")

    # 关键趋势总结
    lines.append("## 📝 关键趋势")
    lines.append("")
    for stage in sorted(report["stage_level"].keys(), key=lambda x: int(x.replace("stage", ""))):
        sdata = report["stage_level"][stage]
        train_total = sdata.get("train", {}).get("total", [])
        val_iou = sdata.get("val", {}).get("iou", [])
        if train_total:
            first = train_total[0]["value"]
            last = train_total[-1]["value"]
            lines.append(f"- **{stage.upper()}** 训练 Total Loss: {_fmt_sci(first)} → {_fmt_sci(last)} (变化比: {last/first:.2f}x)")
        if val_iou:
            first = val_iou[0]["value"]
            last = val_iou[-1]["value"]
            lines.append(f"- **{stage.upper()}** 验证 IoU: {_fmt_float(first)} → {_fmt_float(last)} (变化: {last-first:+.4f})")
    lines.append("")

    lines.append("---")
    lines.append(f"*自动生成于 {meta['exported_at']}*")

    return "\n".join(lines)


def main():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    event_file = find_main_event_file(log_dir)
    print(f"[EXPORT] 读取 TensorBoard event: {event_file}")

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    raw_data = extract_scalars(ea)
    print(f"[EXPORT] 提取了 {len(raw_data)} 个 scalar tags")

    json_report = build_json_report(raw_data, event_file)

    # 保存 JSON
    json_path = os.path.join(log_dir, "training_loss.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"[EXPORT] JSON 已保存: {json_path}")

    # 保存 Markdown
    md_content = build_md_report(json_report)
    md_path = os.path.join(log_dir, "TRAINING_LOG.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[EXPORT] Markdown 已保存: {md_path}")


if __name__ == "__main__":
    main()
