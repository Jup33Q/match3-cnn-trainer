"""model_summary.py - Match3UNet 模型结构优雅打印与分析

用法:
    python -m utils.model_summary [--config config.json] [--board_size 50]

功能:
    1. 树形层级展示模型结构
    2. 按模块分组统计参数量与占比
    3. 高亮 Mamba 层位置
    4. 估算模型大小、FLOPs、显存占用
"""
import argparse
import json
from dataclasses import asdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from config import Match3Config
from models.model import Match3UNet


def format_number(n: int) -> str:
    """格式化大数字"""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def format_shape(shape) -> str:
    """格式化张量形状"""
    if isinstance(shape, torch.Size):
        return str(tuple(shape))
    return str(shape)


def count_conv_flops(module, input_shape, output_shape):
    """估算卷积层FLOPs"""
    if not (hasattr(module, 'kernel_size') and hasattr(module, 'groups')):
        return 0
    if len(input_shape) != 4 or len(output_shape) != 4:
        return 0
    try:
        batch, in_ch, in_h, in_w = input_shape
        batch, out_ch, out_h, out_w = output_shape
        k_h = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        k_w = module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size
        groups = module.groups
        macs_per_pos = (k_h * k_w * in_ch // groups) * out_ch
        total_macs = macs_per_pos * out_h * out_w * batch
        return total_macs * 2
    except Exception:
        return 0


def build_module_tree(model: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Module, int, str]]:
    """
    构建模块树形结构
    返回: [(full_name, module, depth, branch_prefix), ...]
    """
    tree = []
    children = list(model.named_children())
    for idx, (name, module) in enumerate(children):
        is_last = (idx == len(children) - 1)
        branch = "└── " if is_last else "├── "
        indent = prefix + branch
        tree.append((name, module, len(prefix.split("│   ")) - 1 if prefix else 0, indent))

        if len(list(module.children())) > 0:
            child_prefix = prefix + ("    " if is_last else "│   ")
            subtree = build_module_tree(module, child_prefix)
            for sub_name, sub_module, depth, sub_indent in subtree:
                full_name = f"{name}.{sub_name}"
                tree.append((full_name, sub_module, depth + 1, sub_indent))
    return tree


def get_leaf_params(module: nn.Module) -> int:
    """获取叶子模块的参数数量"""
    if len(list(module.children())) == 0:
        return sum(p.numel() for p in module.parameters())
    return 0


def collect_forward_info(model: nn.Module, x: torch.Tensor):
    """收集前向传播的输入/输出信息"""
    info_map = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor) else None
            out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else None
            params = get_leaf_params(module)
            info_map[name] = {
                "in_shape": in_shape,
                "out_shape": out_shape,
                "params": params,
                "module": module,
            }
        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return info_map


def group_params_by_major_module(model: nn.Module) -> Dict[str, int]:
    """按主要模块分组统计参数量"""
    groups = {
        "Stem": 0,
        "Encoders": 0,
        "Bottleneck": 0,
        "Decoders": 0,
        "Final Head": 0,
        "Other": 0,
    }
    for name, module in model.named_modules():
        params = get_leaf_params(module)
        if not params:
            continue
        if name.startswith("stem"):
            groups["Stem"] += params
        elif name.startswith("encoders"):
            groups["Encoders"] += params
        elif name.startswith("bottleneck"):
            groups["Bottleneck"] += params
        elif name.startswith("decoders"):
            groups["Decoders"] += params
        elif name.startswith("final_conv"):
            groups["Final Head"] += params
        else:
            groups["Other"] += params
    return {k: v for k, v in groups.items() if v > 0}


def draw_param_bar(ratio: float, width: int = 30) -> str:
    """绘制参数占比条形图"""
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {ratio*100:.1f}%"


def print_model_tree(model: nn.Module, info_map: dict, total_params: int):
    """优雅打印模型树形结构"""
    print("\n" + "=" * 90)
    print("  Match3UNet 模型架构树")
    print("=" * 90)
    print(f"  {'Module':<45} {'Type':<22} {'Params':>10} {'Output Shape':>18}")
    print("-" * 90)

    tree = build_module_tree(model)

    for name, module, depth, indent in tree:
        info = info_map.get(name, {})
        params = info.get("params", 0)
        out_shape = info.get("out_shape", None)

        module_type = module.__class__.__name__
        is_mamba = "Mamba" in module_type
        is_leaf = len(list(module.children())) == 0

        # 高亮 Mamba 层
        mamba_tag = " 🐍" if is_mamba else ""

        # 只显示叶子节点或重要模块
        display_name = indent + name.split(".")[-1]
        if len(display_name) > 45:
            display_name = display_name[:42] + "..."

        param_str = format_number(params) if params else "-"
        shape_str = format_shape(out_shape) if out_shape else "-"

        # 控制打印深度：只打印到 BasicBlock / Mamba2DLayer / EncoderBlock / DecoderBlock 级别
        # 不再展开 ResBlock 内部的 Conv2d / BN
        important_types = ("BasicBlock", "Mamba2DLayer", "MambaBlock", "EncoderBlock",
                           "DecoderBlock", "Sequential", "ModuleList", "ConvTranspose2d")
        should_print = is_leaf or is_mamba or depth <= 1 or module_type in important_types
        if not should_print:
            continue

        # 对叶子节点（且不是重要容器）汇总参数量到父级，这里只打印重要层级
        if is_leaf and depth > 2 and module_type not in ("MambaBlock", "BasicBlock"):
            continue

        param_str = format_number(params) if params else "-"
        shape_str = format_shape(out_shape) if out_shape else "-"

        line = f"  {display_name:<45} {module_type:<20} {param_str:>10} {shape_str:>18}{mamba_tag}"
        if is_mamba:
            line = f"\033[1;35m{line}\033[0m"  # 紫色高亮
        print(line)

    print("-" * 90)


def print_grouped_params(groups: Dict[str, int], total: int):
    """按模块分组打印参数统计"""
    print("\n" + "=" * 90)
    print("  模块参数量分布")
    print("=" * 90)
    print(f"  {'Module':<20} {'Params':>12} {'Ratio':>8}   {'Bar':<40}")
    print("-" * 90)

    for name, params in sorted(groups.items(), key=lambda x: -x[1]):
        ratio = params / total if total else 0
        print(f"  {name:<20} {format_number(params):>12} {ratio*100:>7.1f}%  {draw_param_bar(ratio)}")

    print("-" * 90)
    print(f"  {'Total':<20} {format_number(total):>12} {'100.0%':>8}")


def print_memory_estimate(total_params: int, layer_info_list: List[dict], batch_size: int, board_size: int):
    """打印显存占用估算"""
    print("\n" + "=" * 90)
    print(f"  显存占用估算 (batch={batch_size}, board={board_size}x{board_size})")
    print("=" * 90)

    model_mem_fp32 = total_params * 4 / 1024 / 1024
    model_mem_fp16 = total_params * 2 / 1024 / 1024
    model_mem_bf16 = model_mem_fp16

    # 激活值估算
    activation_mem = 0.0
    for info in layer_info_list:
        out_shape = info.get("out_shape")
        if out_shape and out_shape != "N/A":
            mem = 4.0
            for dim in out_shape:
                mem *= float(dim)
            activation_mem += mem
    activation_mem = activation_mem / 1024 / 1024

    print(f"  {'模型权重 (FP32)':<25} {model_mem_fp32:>8.2f} MB")
    print(f"  {'模型权重 (BF16/FP16)':<25} {model_mem_bf16:>8.2f} MB")
    print(f"  {'中间激活值 (估算)':<25} {activation_mem:>8.2f} MB")
    print("-" * 90)
    print(f"  {'训练总显存 (FP32)':<25} ~{model_mem_fp32 * 3 + activation_mem * 2:>8.2f} MB  (权重+梯度+优化器状态+激活)")
    print(f"  {'训练总显存 (BF16)':<25} ~{model_mem_bf16 * 3 + activation_mem:>8.2f} MB  (BF16混合精度)")
    print(f"  {'推理显存':<25} ~{model_mem_fp32 + activation_mem * 0.5:>8.2f} MB")


def print_config_summary(config: Match3Config):
    """打印关键配置摘要"""
    print("\n" + "=" * 90)
    print("  模型关键配置")
    print("=" * 90)

    key_configs = [
        ("棋盘尺寸", f"{config.board_size}x{config.board_size}"),
        ("水果种类数", config.num_fruit_types),
        ("基础通道数", config.base_channels),
        ("Encoder 块数", config.num_encoder_blocks),
        ("每 Stage ResBlock", config.blocks_per_stage),
        ("Bottleneck 块数", config.bottleneck_blocks),
        ("使用膨胀卷积", config.use_dilation),
        ("启用 Mamba", f"{'✅ 是' if config.use_mamba else '❌ 否'} (d_state={config.mamba_d_state}, expand={config.mamba_expand})"),
        ("Dropout", config.dropout),
        ("Batch Size", config.batch_size),
        ("训练精度", config.precision.upper()),
        ("学习率", config.learning_rate),
        ("课程学习", f"{'✅ 启用' if config.curriculum_enabled else '❌ 关闭'}"),
    ]

    for key, val in key_configs:
        print(f"  {key:<25} {val}")


def analyze_model_elegantly(model: Match3UNet, input_shape: Tuple[int, ...]):
    """主分析函数：优雅打印模型结构"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "🎮 " * 30)
    print("  Match3UNet 模型结构分析报告")
    print("🎮 " * 30)

    print(f"\n  📊 全局统计")
    print(f"     总参数量:    {format_number(total_params):>10} ({total_params:,})")
    print(f"     可训练参数:  {format_number(trainable_params):>10} ({trainable_params:,})")
    print(f"     模型大小:    {total_params * 4 / 1024 / 1024:.2f} MB (FP32) / {total_params * 2 / 1024 / 1024:.2f} MB (BF16)")

    if getattr(model.cfg, 'use_mamba', False):
        # 估算 Mamba 参数量
        mamba_params = sum(p.numel() for n, p in model.named_parameters() if "mamba" in n.lower() or "bottleneck" in n.lower())
        print(f"     🐍 Mamba 相关: ~{format_number(mamba_params)}")

    # 前向传播收集信息
    x = torch.randn(*input_shape)
    info_map = collect_forward_info(model, x)

    # 构建层级树
    print_model_tree(model, info_map, total_params)

    # 模块分组统计
    groups = group_params_by_major_module(model)
    print_grouped_params(groups, total_params)

    # FLOPs 估算
    layer_info_list = []
    total_flops = 0
    for name, info in info_map.items():
        in_shape = info.get("in_shape")
        out_shape = info.get("out_shape")
        module = info.get("module")
        layer_info_list.append(info)
        if in_shape and out_shape and out_shape != "N/A":
            total_flops += count_conv_flops(module, in_shape, out_shape)

    print("\n" + "=" * 90)
    print("  FLOPs 估算")
    print("=" * 90)
    print(f"  单次前向传播: ~{format_number(total_flops)} FLOPs")
    print(f"  单张推理:     ~{format_number(total_flops // input_shape[0])} FLOPs")

    # 显存估算
    print_memory_estimate(total_params, layer_info_list, input_shape[0], input_shape[2])

    # 配置摘要
    print_config_summary(model.cfg)

    print("\n" + "=" * 90)
    print("  分析完成 ✓")
    print("=" * 90 + "\n")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / 1024 / 1024,
        "estimated_flops": total_flops // input_shape[0],
        "module_groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description="Match3UNet Elegant Model Summary")
    parser.add_argument("--config", type=str, help="JSON config file path")
    parser.add_argument("--board_size", type=int, default=50, help="棋盘尺寸")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--export", type=str, default=None, help="导出JSON报告路径")
    args = parser.parse_args()

    config = Match3Config()
    if args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                if hasattr(config, k):
                    setattr(config, k, v)

    config.board_size = args.board_size

    model = Match3UNet(config)

    # 修复：根据 fruit_embed_dim 或 num_fruit_types 决定输入通道
    in_channels = getattr(config, 'fruit_embed_dim', config.num_fruit_types)
    input_shape = (args.batch_size, in_channels, args.board_size, args.board_size)

    report = analyze_model_elegantly(model, input_shape)

    if args.export:
        report["module_groups"] = {k: v for k, v in report["module_groups"].items()}
        with open(args.export, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n报告已导出至: {args.export}")


if __name__ == "__main__":
    main()
