#!/bin/bash
# =============================================================================
# 上传模型到 ModelScope（只传模型，不传代码）
# =============================================================================
# 说明:
#   在临时目录中准备模型文件 + README，然后推送到 ModelScope。
#   GitHub 仓库保留代码和可视化，ModelScope 只放模型文件。
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMP_DIR="/tmp/match3_modelscope_upload"
MODEL_REPO="https://oauth2:ms-7e72efc7-4da2-4da6-89e4-0b7d6392baf7@www.modelscope.cn/Jup33Q/match3-cnn-trainer.git"

echo "=== 准备 ModelScope 模型文件 ==="

# 清理并创建临时目录
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# 复制模型文件
cp "$REPO_ROOT/checkpoints/match3_unet_epochbest.pt" "$TEMP_DIR/"

# 复制模型配置
cp "$REPO_ROOT/configs/unet_50x50_resume_epoch40.json" "$TEMP_DIR/config.json"

# 复制 ModelScope 专用 README
cp "$REPO_ROOT/README_MODELSCOPE.md" "$TEMP_DIR/README.md"

echo "=== 文件列表 ==="
ls -lh "$TEMP_DIR/"

echo ""
echo "=== 初始化 Git 并推送 ==="
cd "$TEMP_DIR"
git init
git remote add origin "$MODEL_REPO" 2>/dev/null || git remote set-url origin "$MODEL_REPO"
git branch -M main
git add -A
git commit -m "upload model: best checkpoint + config" || echo "(无新变更)"
git push -u origin main --force

echo ""
echo "=== ModelScope 上传完成 ==="
echo "模型地址: https://www.modelscope.cn/models/Jup33Q/match3-cnn-trainer"

# 清理
rm -rf "$TEMP_DIR"
