#!/bin/bash
# =============================================================================
# 配置 Git Push 权限
# =============================================================================
# 提供两种方案：SSH Key（推荐）或 HTTPS Token
# =============================================================================

echo "=== Git Push 权限配置 ==="
echo ""

# 检测是否已有 SSH key
if [ -f ~/.ssh/id_ed25519.pub ] || [ -f ~/.ssh/id_rsa.pub ]; then
    echo "✅ 检测到已有 SSH Key"
    echo ""
    echo "公钥内容如下，请复制添加到 GitHub -> Settings -> SSH and GPG keys -> New SSH key"
    echo "---"
    cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub 2>/dev/null
    echo "---"
    echo ""
else
    echo "🔧 生成 SSH Key..."
    ssh-keygen -t ed25519 -C "jup33q@gmail.com" -N "" -f ~/.ssh/id_ed25519
    echo ""
    echo "✅ SSH Key 已生成"
    echo ""
    echo "请复制以下公钥内容，添加到 GitHub -> Settings -> SSH and GPG keys -> New SSH key"
    echo "---"
    cat ~/.ssh/id_ed25519.pub
    echo "---"
    echo ""
fi

echo "🔧 配置 SSH 并切换远程仓库为 SSH 地址..."
ssh -o StrictHostKeyChecking=accept-new github.com 2>/dev/null || true

cd /home/jup33q/vllm_scripts/super_mirror/match3_cnn_trainer
git remote set-url origin git@github.com:Jup33Q/match3-cnn-trainer.git

echo ""
echo "=== 配置完成 ==="
echo "远程仓库:"
git remote -v
echo ""
echo "⚠️  重要：请先去 GitHub 添加上面的 SSH 公钥，然后再执行:"
echo ""
echo "    cd match3_cnn_trainer"
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""
