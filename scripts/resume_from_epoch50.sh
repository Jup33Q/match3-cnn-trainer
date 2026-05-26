#!/bin/bash
# =============================================================================
# 从 Epoch 50 恢复训练脚本
# =============================================================================
# 说明:
#   当训练到第 50 轮之后因显存/内存过高中断时，使用此脚本从 epoch50 检查点继续。
#   会自动跳过已完成的 epoch (1~50)，从 epoch 51 开始继续训练。
#
# 注意:
#   旧版检查点可能未保存 stage_idx，因此显式指定 --start_stage 2 确保
#   直接进入 50x50 课程学习阶段，避免从 10x10 重新训练。
# =============================================================================

cd "$(dirname "$0")/.."

CHECKPOINT="./checkpoints/match3_unet_epoch50.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 检查点不存在: $CHECKPOINT"
    echo "   请确认 checkpoints/ 目录下有 epoch50 的模型文件。"
    exit 1
fi

echo "=== 从 Epoch 50 恢复训练 ==="
echo "检查点: $CHECKPOINT"
echo "课程阶段: 2 (50×50)"
echo ""

python main.py \
    --mode train \
    --config configs/unet_50x50_resume.json \
    --checkpoint "$CHECKPOINT" \
    --resume \
    --start_stage 2

echo ""
echo "=== 恢复训练完成 ==="
