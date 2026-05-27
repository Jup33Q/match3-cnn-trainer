#!/bin/bash
# =============================================================================
# 从最佳检查点恢复训练脚本 (带 Dropout)
# =============================================================================
# 说明:
#   从最佳模型检查点继续训练，启用 dropout=0.1 防止过拟合，
#   每轮都输出可视化样例。
#
#   课程阶段说明:
#     epoch 1~20  -> 阶段0 (10×10)
#     epoch 21~40 -> 阶段1 (25×25)
#     epoch 41~60 -> 阶段2 (50×50)
#   可从任意阶段恢复，默认从阶段2 (50×50) 开始。
# =============================================================================

cd "$(dirname "$0")/.."

CHECKPOINT="./checkpoints/match3_unet_epochbest.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 检查点不存在: $CHECKPOINT"
    echo "   请确认 checkpoints/ 目录下有训练好的模型文件。"
    exit 1
fi

echo "=== 从最佳检查点恢复训练 (dropout=0.1) ==="
echo "检查点: $CHECKPOINT"
echo "课程阶段: 2 (50×50)"
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
