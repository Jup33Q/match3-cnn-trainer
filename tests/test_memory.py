"""test_memory.py - 实际显存占用测试 (Batch=200, 50x50)

验证在BF16混合精度下，batch_size=200时显存是否控制在8GB以内。
"""
import torch
import numpy as np
from config import Match3Config
from models.model import Match3UNet
from trainer.losses import Match3Loss
from data.data_generator import Match3Dataset
from torch.utils.data import DataLoader


def test_memory():
    print("=" * 60)
    print("显存占用实际测试")
    print("=" * 60)

    config = Match3Config()
    config.batch_size = 200
    config.board_size = 50
    config.precision = "bf16"
    config.num_workers = 0  # 简化测试

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    if not torch.cuda.is_available():
        print("⚠️ 无CUDA设备，跳过显存测试")
        return True

    # 清理显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"基准显存: {base_mem:.2f} GB")

    # 创建模型
    model = Match3UNet(config).to(device)
    model_mem = torch.cuda.memory_allocated() / 1024**3 - base_mem
    print(f"模型加载后: {model_mem:.2f} GB")

    # 创建损失和优化器
    criterion = Match3Loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt_mem = torch.cuda.memory_allocated() / 1024**3 - base_mem - model_mem
    print(f"优化器状态: {opt_mem:.2f} GB")

    # 生成一个batch数据
    ds = Match3Dataset(config, 200, 50)  # 正好1个epoch=1个batch
    loader = DataLoader(ds, batch_size=200, num_workers=0)
    boards, masks = next(iter(loader))
    boards = boards.to(device)
    masks = masks.to(device).unsqueeze(1)
    data_mem = torch.cuda.memory_allocated() / 1024**3 - base_mem - model_mem - opt_mem
    print(f"数据加载: {data_mem:.2f} GB")

    # BF16前向传播
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        preds = model(boards)
        loss = criterion(preds, masks)["total"]

    forward_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n前向传播峰值: {forward_peak:.2f} GB")

    # 反向传播
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    backward_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"反向传播峰值: {backward_peak:.2f} GB")

    # 优化器步骤
    optimizer.step()
    total_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"\n总显存占用: {total_mem:.2f} GB")

    # 验证
    max_allowed = config.max_memory_gb
    passed = total_mem < max_allowed

    print(f"\n限制: {max_allowed} GB")
    print(f"结果: {'✅ 通过' if passed else '❌ 超限'}")

    # 估算训练速度
    print(f"\n[性能估算]")
    print(f"  单batch前向: ~{forward_peak - base_mem:.2f} GB")
    print(f"  完整训练(含梯度): ~{total_mem:.2f} GB")
    print(f"  安全余量: {max_allowed - total_mem:.2f} GB")

    print("=" * 60)
    return passed


if __name__ == "__main__":
    import sys
    ok = test_memory()
    sys.exit(0 if ok else 1)
