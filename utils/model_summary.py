"""model_summary.py - Match3UNet 模型结构与参数计算器

用法:
    python model_summary.py [--config config.json] [--board_size 50]

功能:
    1. 构建 Match3UNet 模型
    2. 逐层分析参数量、输出形状
    3. 计算总参数量、可训练/不可训练参数
    4. 估算模型大小 (MB)
    5. 估算前向传播 FLOPs
    6. 50x50 棋盘下的显存占用预估
"""
import argparse
import json
from dataclasses import asdict

from config import Match3Config
from models.model import Match3UNet


def format_number(n: int) -> str:
    """格式化大数字"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def count_conv_flops(module, input_shape, output_shape):
    """估算卷积层FLOPs"""
    # FLOPs ≈ 2 * H_out * W_out * C_out * K_h * K_w * C_in
    if hasattr(module, 'kernel_size') and hasattr(module, 'groups'):
        batch, in_ch, in_h, in_w = input_shape
        batch, out_ch, out_h, out_w = output_shape
        k_h = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        k_w = module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size
        groups = module.groups
        # MACs per output position
        macs_per_pos = (k_h * k_w * in_ch // groups) * out_ch
        total_macs = macs_per_pos * out_h * out_w * batch
        return total_macs * 2  # FLOPs ≈ 2 * MACs
    return 0


def analyze_model(model, input_shape=(1, 6, 50, 50)):
    """逐层分析模型结构与参数"""
    import torch

    print("=" * 80)
    print("Match3UNet 模型结构与参数分析报告")
    print("=" * 80)

    # 基本统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params

    print(f"\n[全局统计]")
    print(f"  总参数量:       {format_number(total_params):>12} ({total_params:,})")
    print(f"  可训练参数:     {format_number(trainable_params):>12} ({trainable_params:,})")
    print(f"  不可训练参数:   {format_number(non_trainable):>12} ({non_trainable:,})")
    print(f"  模型大小(FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  模型大小(FP16): {total_params * 2 / 1024 / 1024:.2f} MB")

    # 逐层分析
    print(f"\n[逐层结构分析] 输入: {input_shape}")
    print("-" * 80)
    print(f"{'Layer Name':<35} {'Type':<20} {'Params':>12} {'Output Shape':>20}")
    print("-" * 80)

    x = torch.randn(*input_shape)
    hooks = []
    layer_info = []

    def hook_fn(name, module):
        def fn(module, input, output):
            params = sum(p.numel() for p in module.parameters())
            out_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else "N/A"
            layer_type = module.__class__.__name__
            layer_info.append({
                "name": name,
                "type": layer_type,
                "params": params,
                "out_shape": out_shape,
                "module": module,
                "input_shape": tuple(input[0].shape) if input and isinstance(input[0], torch.Tensor) else None
            })
        return fn

    # 注册hook
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and name:  # 叶子节点
            hooks.append(module.register_forward_hook(hook_fn(name, module)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    # 打印
    total_flops = 0
    for info in layer_info:
        print(f"{info['name']:<35} {info['type']:<20} {format_number(info['params']):>12} {str(info['out_shape']):>20}")
        if info['input_shape'] and info['out_shape'] != "N/A":
            total_flops += count_conv_flops(info['module'], info['input_shape'], info['out_shape'])

    # 显存估算
    print(f"\n[显存占用预估] (batch_size={input_shape[0]}, board={input_shape[2]}x{input_shape[3]})")
    print("-" * 80)
    
    # 模型权重显存
    model_mem_fp32 = total_params * 4 / 1024 / 1024
    model_mem_fp16 = total_params * 2 / 1024 / 1024
    
    # 激活值显存估算 (简化: 每层输出保留到backward)
    # 粗略估算: 前向传播中所有中间特征图的总和
    activation_mem = 0.0
    for info in layer_info:
        if info['out_shape'] != "N/A" and isinstance(info['out_shape'], tuple):
            mem = 4.0
            for dim in info['out_shape']:
                mem *= float(dim)
            activation_mem += mem
    activation_mem = activation_mem / 1024.0 / 1024.0

    model_mem_bf16 = total_params * 2 / 1024 / 1024  # BF16同FP16大小
    print(f"  模型权重 (FP32):     {model_mem_fp32:.2f} MB")
    print(f"  模型权重 (BF16):     {model_mem_bf16:.2f} MB")
    print(f"  中间激活值 (估算):   {activation_mem:.2f} MB")
    print(f"  训练总显存 (FP32):   ~{model_mem_fp32 * 3 + activation_mem * 2:.2f} MB  (权重+梯度+优化器状态+激活)")
    print(f"  训练总显存 (BF16):   ~{model_mem_bf16 * 3 + activation_mem:.2f} MB  (BF16混合精度)")
    print(f"  推理显存:            ~{model_mem_fp32 + activation_mem * 0.5:.2f} MB")

    print(f"\n[FLOPs 估算]")
    print(f"  单次前向传播:        ~{format_number(total_flops)} FLOPs")
    print(f"  单张50x50推理:       ~{format_number(total_flops // input_shape[0])} FLOPs")

    # 配置信息
    print(f"\n[当前配置]")
    cfg_dict = asdict(model.cfg)
    for k, v in cfg_dict.items():
        print(f"  {k:<30} {v}")

    print("=" * 80)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / 1024 / 1024,
        "estimated_flops": total_flops // input_shape[0]
    }


def main():
    parser = argparse.ArgumentParser(description="Match3UNet Model Summary")
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

    # 临时修改board_size用于分析
    config.board_size = args.board_size

    model = Match3UNet(config)
    input_shape = (args.batch_size, config.num_fruit_types, args.board_size, args.board_size)
    report = analyze_model(model, input_shape)

    if args.export:
        with open(args.export, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n报告已导出至: {args.export}")


if __name__ == "__main__":
    main()
