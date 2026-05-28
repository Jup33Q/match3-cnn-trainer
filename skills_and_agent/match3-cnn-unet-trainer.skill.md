<!--
  Skill: match3-cnn-unet-trainer
  Description: 基于U-Net的轻量级三消Pattern识别模型训练规范，使用大卷积核和膨胀卷积捕获局部连续性
  Version: 1.0.0
  Author: AI Skill Designer
  Stack: PyTorch >=2.0.0, Python >=3.9
  Dependencies: torch>=2.0.0, torchvision>=0.15.0, numpy>=1.24.0, tqdm>=4.65.0, tensorboard>=2.13.0
  Base Skills: match3-pattern-core, match3-data-generation, match3-training-foundation
-->

# match3-cnn-unet-trainer

## 1. 概述

### 1.1 架构选择动机

U-Net架构天然适合三消（Match-3）Pattern识别任务，原因如下：

| 特性 | 三消Pattern需求 | U-Net匹配度 |
|------|----------------|------------|
| 局部连续性 | 消除块需在行/列连续 | Encoder逐层下采样聚合上下文 |
| 精确位置 | 输出需像素级mask | Decoder上采样恢复空间分辨率 |
| 多尺度模式 | L/T/Cross形状大小不一 | 多层Skip Connection融合多尺度特征 |
| 轻量高效 | 需实时推理 | 深度可分离卷积 + 可选膨胀减少参数量 |

**核心设计**: Encoder-Decoder + Skip Connection。Encoder捕获"连续相同水果"的语义特征，Decoder通过Skip Connection保留精确棋盘位置信息，最终输出像素级消除mask。

### 1.2 输入输出规范

- **输入**: 正n边形顶点3通道棋盘 `(B, max_fruit_types+1, H, W)`，前16个通道为fruit类型的One-Hot编码，第17个通道为valid_mask（可选）
- **输出**: Sigmoid激活的mask `(B, 1, H, W)`，值域`[0,1]`表示每个格子属于消除Pattern的概率

---

## 2. 架构设计规范

### 2.1 整体结构

```
Input: (B, max_fruit_types+1, H, W)  # 16 One-Hot + 1 valid_mask
  ↓
[Initial Conv: 7x7 kernel] ──→ 捕获局部连续性
  ↓
[Encoder Block 1] ──→ downsample ──→ skip_1 ──→ (B, C,   H/2, W/2)
[Encoder Block 2] ──→ downsample ──→ skip_2 ──→ (B, 2C,  H/4, W/4)
[Encoder Block 3] ──→ downsample ──→ skip_3 ──→ (B, 4C,  H/8, W/8)
[Encoder Block 4] ──→ downsample ──→ skip_4 ──→ (B, 8C,  H/16, W/16)
  ↓
[Bottleneck: 2×Conv + optional Dilated Conv]
  ↓
[Decoder Block 4] ←── concat skip_4 ←── upsample ──→ (B, 8C, H/16, W/16)
[Decoder Block 3] ←── concat skip_3 ←── upsample ──→ (B, 4C, H/8,  W/8)
[Decoder Block 2] ←── concat skip_2 ←── upsample ──→ (B, 2C, H/4,  W/4)
[Decoder Block 1] ←── concat skip_1 ←── upsample ──→ (B, C,  H/2,  W/2)
  ↓
[Final Upsample]
  ↓
[Output Head: 1x1 Conv + Sigmoid]
  ↓
Output: (B, 1, H, W)
```

### 2.2 各层详细规范

#### 2.2.1 首层卷积 (Initial Conv)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_kernel_size` | `7` | 大卷积核捕获局部连续性 |
| `initial_stride` | `1` | 保持分辨率 |
| `initial_padding` | `3` | Same padding |
| `input_channels` | `max_fruit_types+1=17` | 正n边形顶点3通道 + valid_mask |
| `base_channels` | `32` | 首层输出通道数 |

> **设计原理**: 三消规则依赖相邻格子的水果类型关系。7×7卷积核覆盖足够大的邻域，让模型在首层即感知"3个以上连续相同水果"的局部模式。

#### 2.2.2 Encoder 规范

- **层数**: `num_encoder_blocks` (默认 `4`)
- **每块结构**:
  ```
  ConvBlock(in_ch, out_ch) → MaxPool2d(2)
  ```
- **通道翻倍**: 每层输出通道翻倍 `C → 2C → 4C → 8C`
- **下采样**: `MaxPool2d(kernel_size=2, stride=2)`

#### 2.2.3 膨胀卷积选项

当 `use_dilation=True` 时，Encoder各层使用膨胀卷积：

| Encoder层 | 膨胀率 | 有效感受野 |
|-----------|--------|-----------|
| Block 1 | `dilation=1` | 3×3 |
| Block 2 | `dilation=2` | 5×5 |
| Block 3 | `dilation=4` | 9×9 |
| Block 4 | `dilation=8` | 17×17 |

> **适用场景**: 大棋盘(50×50+)时启用，在不增加参数的情况下扩大感受野。

#### 2.2.4 Bottleneck 规范

```
ConvBlock(in_ch=8C, mid_ch=16C, out_ch=8C)
```

- 双层卷积 + BatchNorm + ReLU
- 可选SE注意力模块 (`use_se=True`)

#### 2.2.5 Decoder 规范

- **层数**: 与Encoder对称，4层上采样
- **每块结构**:
  ```
  Upsample(scale_factor=2) → ConvBlock(in_ch, out_ch) → Concat(skip)
  ```
- **Skip Connection**: 与对应Encoder层输出通道拼接

#### 2.2.6 输出头规范

| 组件 | 配置 |
|------|------|
| 卷积层 | `Conv2d(base_channels, 1, kernel_size=1)` |
| 激活函数 | `Sigmoid()` |
| 输出形状 | `(B, 1, H, W)` |
| 值域 | `[0, 1]` — 消除概率 |

---

## 3. 配置接口规范

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Match3UNetConfig:
    # === 架构参数 ===
    num_fruit_types: int = 6      # 水果种类数（如6）
    fruit_embed_dim: int = 32     # (已废弃) 原正n边形顶点编码维度，现统一使用One-Hot
    max_fruit_types: int = 16     # 正n边形顶点3通道数
    base_channels: int = 32       # 首层通道数
    num_encoder_blocks: int = 4   # Encoder/Decoder层数

    # === 卷积参数 ===
    initial_kernel_size: int = 7  # 首层大卷积核
    use_dilation: bool = False    # 是否启用膨胀卷积
    dilation_rates: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    use_se: bool = False          # SE注意力模块

    # === 训练参数 ===
    learning_rate: float = 1e-3
    batch_size: int = 8
    num_epochs: int = 100
    device: str = "cuda"

    # === 数据参数 ===
    board_height: int = 20        # 棋盘高度
    board_width: int = 20         # 棋盘宽度

    # === 课程学习 ===
    curriculum_stages: List[int] = field(default_factory=lambda: [10, 25, 50, -1, -3, -2])  # -1=正方形随机过渡, -3=长方形随机, -2=stage5全面泛化

    def build_model(self):
        # 返回 Match3UNet 实例 (stem input_ch = getattr(config, 'max_fruit_types', config.num_fruit_types) + (1 if use_valid_mask else 0))
        pass
```

---

## 4. 后处理规则约束层

### 4.1 enforce_match3_rules 规范

输出mask必须经过规则约束层，确保只识别合法的Match-3消除模式：

```python
def enforce_match3_rules(mask: Tensor, board: Tensor) -> Tensor:
    """
    后处理：BFS连通分量检测 + 直线连续性验证
    
    Args:
        mask: (B, 1, H, W) Sigmoid输出
        board: (B, num_types, H, W) one-hot棋盘
    
    Returns:
        filtered_mask: (B, 1, H, W) 规则过滤后的mask
    """
    # 1. 二值化阈值
    binary = (mask > threshold).float()
    # 2. BFS连通分量检测 — 找连通区域
    components = bfs_connected_components(binary)
    # 3. 直线连续性验证 — 同行/同列连续>=3
    valid = validate_line_continuity(components, min_length=3)
    # 4. 水果类型一致性验证
    valid = validate_fruit_consistency(valid, board)
    return valid * mask
```

### 4.2 约束规则说明

| 规则 | 验证方法 | 目的 |
|------|---------|------|
| 连通分量 | BFS/Flood Fill | 消除块必须相邻连通 |
| 直线连续性 | 行/列扫描 | 至少3个同类型在同一直线 |
| 类型一致性 | 与board对比 | mask覆盖的格子水果类型一致 |
| 最小长度 | `min_length >= 3` | Match-3最低要求 |

---

## 5. 架构变体建议

### 5.1 MobileNetV3 轻量版

| 对比项 | U-Net(本Skill) | MobileNetV3版 |
|--------|---------------|---------------|
| 骨干 | 标准卷积 | 深度可分离卷积 |
| 参数量 | ~2M | ~500K |
| 速度 | 中等 | 快3-5x |
| 精度 | 高 | 略降1-2% IoU |
| 适用 | 原型验证 | 移动端部署 |

**改动点**: 将Encoder中的`ConvBlock`替换为`InvertedResidual`块，使用h-swish激活。

### 5.2 SegFormer 语义分割版

| 对比项 | U-Net(本Skill) | SegFormer版 |
|--------|---------------|-------------|
| 骨干 | CNN Encoder | MiT (Mix Transformer) |
| 解码器 | 对称上采样 | MLP轻量解码 |
| 全局建模 | 有限 | 自注意力 |
| 适用 | 局部模式 | 大棋盘全局模式 |

---

## 6. 使用场景推荐

| 场景 | 推荐配置 | 理由 |
|------|---------|------|
| 快速原型验证 | `base_channels=32, use_dilation=False` | 训练快，收敛稳定 |
| 资源受限环境 | `base_channels=16, num_encoder_blocks=3` | 显存<2GB |
| 大棋盘(50x50+) | `use_dilation=True, base_channels=64` | 感受野覆盖全局 |
| 高精度需求 | `use_se=True, base_channels=64` | SE注意力提升细节 |

---

## 7. 性能预期

| 棋盘尺寸 | 参数量 | 显存(训练) | 显存(推理) | IoU(训练集) | IoU(测试集) |
|---------|--------|-----------|-----------|------------|------------|
| 20×20 | ~1.2M | ~1.5 GB | ~0.5 GB | 0.96+ | 0.93+ |
| 50×50 | ~2.1M | ~3.2 GB | ~1.1 GB | 0.94+ | 0.90+ |
| 100×100 | ~2.1M | ~6.8 GB | ~2.4 GB | 0.91+ | 0.86+ |

> **注**: 上述数据基于 `base_channels=32, num_encoder_blocks=4, batch_size=8` 配置，实际性能因硬件和数据分布而异。

---

## 8. 调优建议

### 8.1 首层卷积核选择

| 棋盘尺寸 | 推荐kernel_size | 理由 |
|---------|----------------|------|
| 10×10以下 | 5 | 棋盘小，过大卷积浪费 |
| 10×10 ~ 50×50 | 7 | 平衡感受野和参数量 |
| 50×50以上 | 7 + dilation | 大卷积+膨胀捕获长距离依赖 |

### 8.2 通道数调优

- `base_channels=16`: 最轻量，适合嵌入式
- `base_channels=32`: 平衡选择（推荐默认值）
- `base_channels=64`: 高精度，显存允许时选择

### 8.3 损失函数建议

- **Mask预测**: `BCEWithLogitsLoss` 或 `FocalLoss + DiceLoss` 组合
- **权重**: 正样本(消除格)加权，解决类别不平衡

### 8.4 数据增强

- 随机旋转90度倍数（保持网格对齐）
- 水果类型随机置换（标签同步变换）
- 棋盘边缘随机padding

---

## 9. 依赖与安装

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 tqdm>=4.65.0 tensorboard>=2.13.0
```

### 9.1 基础Skill依赖

| Skill | 用途 |
|-------|------|
| `match3-pattern-core` | Pattern定义与规则验证 |
| `match3-data-generation` | 棋盘数据生成与增强 |
| `match3-training-foundation` | 训练循环、损失函数、优化器基础 |

---

## 10. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2024-XX-XX | 初始版本：U-Net + 膨胀卷积 + 规则约束层 |

---

<!-- file-tags: skill, match3, cnn, unet, pytorch, pattern-recognition -->
