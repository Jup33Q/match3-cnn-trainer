#!/bin/bash
# =============================================================================
# 低显存模式：从最佳检查点恢复训练 (带 Dropout)
# =============================================================================
# 说明:
#   如果在 50×50 阶段出现 OOM (显存不足)，可使用此脚本通过以下方式
#   降低显存占用:
#     1. PYTORCH_CUDA_ALLOC_CONF: 优化 CUDA 显存分配器碎片管理
#     2. expandable_segments: 启用显存段扩展，减少碎片
#     3. dropout=0.1 防止过拟合
#
#   如需修改起始轮数，请更改下面的 CHECKPOINT 变量。
# =============================================================================

cd "$(dirname "$0")/.."

CHECKPOINT="./checkpoints/match3_unet_epochbest.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 检查点不存在: $CHECKPOINT"
    exit 1
fi

# 显存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

echo "=== 低显存模式：从最佳检查点恢复训练 (dropout=0.1) ==="
echo "检查点: $CHECKPOINT"
echo "CUDA 显存优化: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

python main.py \
    --mode train \
    --config configs/unet_50x50_resume_epoch40.json \
    --checkpoint "$CHECKPOINT" \
    --resume \
    --start_stage 2

echo ""
echo "=== 训练完成 ==="
