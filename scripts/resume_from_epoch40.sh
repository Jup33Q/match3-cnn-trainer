#!/bin/bash
# =============================================================================
# 从 Epoch 40 恢复训练脚本 (带 Dropout)
# =============================================================================
# 说明:
#   从 epoch40 检查点继续训练，启用 dropout=0.1 防止过拟合，
#   每轮都输出可视化样例。
#   会自动跳过已完成的 epoch (1~40)，从 epoch 41 开始继续训练。
#
#   课程阶段说明:
#     epoch 1~20  -> 阶段0 (10×10)
#     epoch 21~40 -> 阶段1 (25×25)
#     epoch 41~60 -> 阶段2 (50×50)
#   因此从 epoch40 恢复后，下一轮 (41) 即进入 50×50 阶段。
# =============================================================================

cd "$(dirname "$0")/.."

CHECKPOINT="./checkpoints/match3_unet_epoch40.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 检查点不存在: $CHECKPOINT"
    echo "   请确认 checkpoints/ 目录下有 epoch40 的模型文件。"
    exit 1
fi

echo "=== 从 Epoch 40 恢复训练 (dropout=0.1) ==="
echo "检查点: $CHECKPOINT"
echo "课程阶段: 2 (50×50) 从 epoch41 开始"
echo "可视化: 每轮输出"
echo ""

python main.py \
    --mode train \
    --config configs/unet_50x50_resume_epoch40.json \
    --checkpoint "$CHECKPOINT" \
    --resume \
    --start_stage 2

echo ""
echo "=== 恢复训练完成 ==="
