#!/bin/bash
# =============================================================================
# 从最佳检查点恢复训练并启用 Dropout
# =============================================================================
# 说明:
#   从最佳模型检查点加载权重，恢复训练并启用 dropout=0.1。
#   适用于在课程学习中后段（如 25×25 或 50×50 阶段）启用正则化防止过拟合。
#
#   默认从 Stage 1 (25×25) 开始，可通过命令行参数指定阶段:
#     bash scripts/resume_with_dropout.sh 2   # 从 Stage 2 (50×50) 开始
#
#   课程阶段与 epoch 对应关系:
#     Stage 0 (10×10) : epoch 1~20
#     Stage 1 (25×25) : epoch 21~40   <- 默认，约第 30 epoch 阶段
#     Stage 2 (50×50) : epoch 41~60
# =============================================================================

cd "$(dirname "$0")/.."

CHECKPOINT="./checkpoints/match3_unet_epochbest.pt"
START_STAGE="${1:-1}"  # 默认 stage 1，可通过第一个参数覆盖

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 检查点不存在: $CHECKPOINT"
    echo "   请确认 checkpoints/ 目录下有训练好的模型文件。"
    exit 1
fi

case "$START_STAGE" in
    0) STAGE_LABEL="10×10" ;;
    1) STAGE_LABEL="25×25" ;;
    2) STAGE_LABEL="50×50" ;;
    *) echo "⚠️ 警告: 未知的 stage $START_STAGE，将尝试继续"; STAGE_LABEL="自定义" ;;
esac

echo "=== 从最佳检查点恢复训练 (启用 Dropout 0.1) ==="
echo "检查点: $CHECKPOINT"
echo "课程阶段: $START_STAGE ($STAGE_LABEL)"
echo "Dropout: 0.1"
echo ""

python main.py \
    --mode train \
    --config configs/unet_50x50_resume_epoch40.json \
    --checkpoint "$CHECKPOINT" \
    --resume \
    --start_stage "$START_STAGE"

echo ""
echo "=== 恢复训练完成 ==="
