"""PyTorch 原生模型架构可视化工具演示

支持方式:
1. torchinfo    - 表格形式模型摘要 (推荐)
2. TensorBoard  - 计算图可视化
3. Netron       - 交互式模型查看器 (导出 ONNX 后)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from config import Match3Config
from models.model import Match3UNet

# ============================================================
# 方式1: torchinfo - 表格形式模型摘要 (最推荐)
# ============================================================
print("=" * 70)
print("方式1: torchinfo - 表格形式模型摘要")
print("=" * 70)

try:
    from torchinfo import summary

    cfg = Match3Config()
    model = Match3UNet(cfg)

    # 支持多种输入尺寸 (stage1~stage5 的不同尺寸)
    test_configs = [
        ("Stage1: 10×10", (1, cfg.fruit_embed_dim, 10, 10)),
        ("Stage3: 50×50", (1, cfg.fruit_embed_dim, 50, 50)),
        ("Batch=80, 50×50", (80, cfg.fruit_embed_dim, 50, 50)),
    ]

    for name, input_size in test_configs:
        print(f"\n>>> {name}, input_shape={input_size}")
        summary(
            model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            col_width=18,
            row_settings=["var_names"],
            depth=4,
            device="cpu"
        )
except ImportError:
    print("torchinfo 未安装，请运行: pip install torchinfo")


# ============================================================
# 方式2: TensorBoard - 计算图可视化
# ============================================================
print("\n" + "=" * 70)
print("方式2: TensorBoard - 计算图可视化")
print("=" * 70)

try:
    from torch.utils.tensorboard import SummaryWriter

    cfg = Match3Config()
    model = Match3UNet(cfg)
    model.eval()

    dummy_input = torch.randn(1, cfg.fruit_embed_dim, 50, 50)

    log_dir = "./logs/model_graph"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    with torch.no_grad():
        writer.add_graph(model, dummy_input)

    writer.close()
    print(f"计算图已保存到: {log_dir}")
    print("查看方式: tensorboard --logdir=./logs/model_graph")
    print("在浏览器中打开 GRAPHS 标签页即可查看模型计算图")
except Exception as e:
    print(f"TensorBoard 可视化失败: {e}")


# ============================================================
# 方式3: Netron - 交互式模型查看器
# ============================================================
print("\n" + "=" * 70)
print("方式3: Netron - 交互式模型查看器")
print("=" * 70)

try:
    import netron

    cfg = Match3Config()
    model = Match3UNet(cfg)
    model.eval()

    dummy_input = torch.randn(1, cfg.fruit_embed_dim, 50, 50)

    # 导出 ONNX
    onnx_path = os.path.join(os.path.dirname(__file__), '..', "match3_unet_visualization.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=11,
    )
    print(f"ONNX 模型已导出: {onnx_path}")
    print(f"模型大小: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")

    # 启动 Netron (非阻塞)
    print("\n启动 Netron 交互式查看器...")
    print("如果下面卡住，请按 Ctrl+C 退出，然后手动运行: netron ./match3_unet_visualization.onnx")
    netron.start(onnx_path, browse=False, address=('localhost', 8080))
    print("Netron 已在 http://localhost:8080 启动")
    print("请在浏览器中访问该地址查看交互式模型结构")

except ImportError:
    print("netron 未安装，请运行: pip install netron")
except Exception as e:
    print(f"Netron 可视化失败: {e}")


# ============================================================
# 方式4: PyTorch 内置 print(model) - 最基础
# ============================================================
print("\n" + "=" * 70)
print("方式4: PyTorch 内置 print(model) - 最基础")
print("=" * 70)

cfg = Match3Config()
model = Match3UNet(cfg)
print(model)


# ============================================================
# 方式5: 自定义模块统计
# ============================================================
print("\n" + "=" * 70)
print("方式5: 自定义模块统计")
print("=" * 70)

cfg = Match3Config()
model = Match3UNet(cfg)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数量:     {total / 1e6:.2f} M")
print(f"可训练参数:   {trainable / 1e6:.2f} M")
print(f"不可训练参数: {(total - trainable) / 1e6:.2f} M")

print("\n各模块参数统计:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:20s} {params / 1e6:8.2f} M  ({params / total * 100:5.1f}%)")

print("\n各层参数详细统计 (Top 10):")
layer_params = []
for name, param in model.named_parameters():
    layer_params.append((name, param.numel()))
layer_params.sort(key=lambda x: x[1], reverse=True)
for name, num in layer_params[:10]:
    print(f"  {name:50s} {num / 1e6:8.3f} M")

print("\n" + "=" * 70)
print("所有可视化方式演示完成!")
print("=" * 70)
