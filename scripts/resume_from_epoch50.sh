#!/bin/bash
# =============================================================================
# 从最佳检查点恢复训练脚本
# =============================================================================
# 说明:
#   当训练中断时，使用此脚本从最佳检查点继续。
#   会自动加载模型权重、优化器状态、学习率调度器和已训练 epoch 数。
#
#   旧版检查点可能未保存 stage_idx，因此显式指定 --start_stage 2 确保
#   直接进入 50x50 课程学习阶段，避免从 10x10 重新训练。
# =============================================================================

cd "$(dirname "$0")/.."

CHECKPOINT="./checkpoints/match3_unet_epochbest.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 检查点不存在: $CHECKPOINT"
    echo "   请确认 checkpoints/ 目录下有训练好的模型文件。"
    exit 1
fi

echo "=== 从最佳检查点恢复训练 ==="
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
