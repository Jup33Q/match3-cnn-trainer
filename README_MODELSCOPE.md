# Match-3 U-Net 模型 (50×50)

> 基于 **Soft-Routed Parallel CNN U-Net** 的三消（Match-3）Pattern 识别模型。

## 模型信息

| 项目 | 详情 |
|------|------|
| 架构 | Soft-Routed Parallel CNN U-Net (v3) |
| Encoder | 5 层，每 stage 2× BasicBlock (ResNet 风格) |
| Bottleneck | Soft-Router + 3 路并行 CNN 分支 (不同 dilation) + 加权求和 |
| Decoder | 5 层，每 stage 2× BasicBlock + Bilinear 上采样 |
| Output | Transformer Spatial Block (Pre-LN MSA + FFN) + 1×1 Conv logits |
| 输入尺寸 | 50×50 棋盘 |
| 输入通道 | 3~4 通道 (cos, sin, 1, 可选 valid_mask) |
| 水果编码 | 正 n 边形顶点编码，支持动态种类数 |
| 水果种类 | 6 种 (训练时)，支持推理时动态种类 |
| 参数量 | **~176M** |
| 精度 | BF16 混合精度训练 |
| 输出 | 消除位置 Mask (logits，无 sigmoid) |

## 模型文件

| 文件 | 说明 |
|------|------|
| `match3_unet_epochfinal.pt` | 最终训练模型（推荐） |
| `match3_unet_epoch120.pt` | Stage 5 最终 Checkpoint |
| `config.json` | 模型配置文件 |

## 模型架构

![Architecture Diagram](./architecture_diagram.png)

### 核心组件

| 组件 | 说明 |
|------|------|
| **Stem** | 7×7 大卷积核 + GroupNorm + SiLU |
| **BasicBlock** | GN → 3×3 Conv → SiLU → GN → 3×3 Conv → SEBlock → LayerScale → DropPath + 残差 |
| **EncoderBlock** | N× BasicBlock + MaxPool2d，Skip Connection 输出 |
| **DecoderBlock** | Bilinear Upsample + 1×1 Conv + Concat Skip + N× BasicBlock |
| **SoftRouter** | AdaptiveAvgPool + 1×1 Conv → GN → SiLU → 1×1 Conv，生成各分支 softmax 权重 |
| **SoftRouterCNNBranch** | 1×1 扩增 → N× BasicBlock → 1×1 缩减 + 残差缩放 → SiLU |
| **RouterSum** | Softmax 加权求和各分支输出 |
| **TransformerSpatialBlock** | 2D→序列 → Pre-LN MSA → Pre-LN FFN → 序列→2D + 残差 |
| **SEBlock** | Squeeze-and-Excitation 通道注意力 |
| **LayerScale** | 可学习残差缩放 (初始 1e-5) |
| **DropPath** | Stochastic Depth，按深度线性递增 drop 概率 |

### 激活函数与归一化

- **激活函数**: SiLU (Swish)，避免 inplace ReLU 导致的 segfault
- **归一化**: GroupNorm (Encoder/Decoder/Bottleneck) + LayerNorm (Transformer)
- **初始化**: Xavier Normal (Glorot)

## 支持的 Pattern

- 横向/纵向 3/4/5 连 (H3~H5, V3~V5)
- L 型、T 型、十字型
- Stage 4~6 支持特殊负样本训练（随机尺寸、随机颜色、加长 match）

## 使用方式

```python
import torch
from models.model import Match3UNet
from config import Match3Config

config = Match3Config()
config.dropout = 0.0
model = Match3UNet(config)

# 加载最终模型
checkpoint = torch.load("match3_unet_epochfinal.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 推理 (输入: [B, 4, H, W] = [batch, channels, 50, 50])
# channels: [cos, sin, 1, valid_mask]
logits = model(x)  # [B, 1, 50, 50]
```

## 训练配置

| 配置项 | 值 |
|--------|-----|
| Batch size | 40 (Stage 1~2 使用 100) |
| Dropout | 0.0 |
| Drop Path | 0.0 (默认可配) |
| 学习率 | 1e-3 (Cosine + Warmup 5 epochs) |
| 权重衰减 | 1e-4 |
| 梯度裁剪 | 1.0 |
| 损失函数 | Dice (0.5) + Focal (0.3) + Boundary (0.2) |
| Focal 参数 | γ=2.0, α=0.25 |
| 课程学习 | 10×10 → 25×25 → 50×50 → 随机正方形 → 随机长方形 → Stage5 强化 (6 stages) |
| 每 stage epoch | 20 |

### Bottleneck 分支配置

| 分支 | 扩张倍数 | Block 数 | Dilation |
|------|---------|---------|----------|
| Branch 0 | 1 | 1 | 1 |
| Branch 1 | 1 | 1 | 2 |
| Branch 2 | 1 | 1 | 4 |

## 训练结果摘要

| Stage | Epochs | Train Total Loss | Val IoU |
|-------|--------|------------------|---------|
| Stage 2 | 41–60 | 9.46e-06 → 1.49e-04 | 0.4680 |
| Stage 3 | 61–65 | 7.09e-03 → 4.79e-04 | 0.4543 → 0.4520 |
| Stage 5 | 101–120 | 2.31e-01 → 4.18e-03 | 0.1810 → **0.3750** |

> Stage 5 为负样本强化训练阶段（随机长方形棋盘 + 随机颜色种类 + match 长度 5~8），IoU 从 0.1810 提升至 0.3750，Total Loss 下降约 98%。

## 相关链接

- 代码仓库: https://github.com/Jup33Q/match3-cnn-trainer
- 训练日志: 见 GitHub 仓库 `logs/TRAINING_LOG.md`
