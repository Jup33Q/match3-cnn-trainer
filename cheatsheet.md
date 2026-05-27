# Match3UNet CNN-RNN-Transformer 架构 Cheatsheet

## 1. 架构总览

```
Input: (B, fruit_embed_dim, H, W)  [RoPE 编码]
  ↓
[Stem: Conv7×7 → BN → SiLU]
  ↓
[Encoder × 5]  →  skips[0..4]
  ↓
[Bottleneck: CNN-RNN Hybrid]
    ├── CNN path: 1× BasicBlock (空间特征精炼)
    └── RNN path:
        ├── Row-GRU: (B, C, H, W) → (B×H, W, C) → Bi-GRU → (B, C, H, W)
        ├── Col-GRU: (B, C, H, W) → (B×W, H, C) → Bi-GRU → (B, C, H, W)
        └── 融合: CNN_out + (Row_out + Col_out) * 0.5
    └── 残差: x + gamma * fused
  ↓
[Decoder × 5]  ←  skips[4..0]
  ↓
[Transformer Output Block]
    ├── 2D→序列: (B, C, H, W) → (B, H*W, C)
    ├── Multi-Head Self-Attention (num_heads=8)
    ├── FFN (Linear → GELU → Linear)
    ├── 2D恢复: (B, H*W, C) → (B, C, H, W)
    └── 残差 + LayerNorm
  ↓
[Output Head: Conv1×1] → (B, 1, H, W) logits
```

---

## 2. 新旧代码对照

### 2.1 Bottleneck (Mamba → CNN-RNN)

**旧 (Mamba2DLayer)**:
```python
from .mamba_layer import Mamba2DLayer

if config.use_mamba:
    for i in range(n_bot):
        if i % 2 == 1:  # 奇数位置放 Mamba
            self.bottleneck.append(Mamba2DLayer(
                ch, config.mamba_d_state, config.mamba_d_conv, config.mamba_expand
            ))
        else:
            self.bottleneck.append(BasicBlock(ch, ch))
```

**新 (CNNRNNBottleneck)**:
```python
if config.use_cnn_rnn:
    for i in range(n_bot):
        if i == 0:
            self.bottleneck.append(CNNRNNBottleneck(
                ch, config.rnn_hidden_ratio, config.num_gru_layers, config.rnn_dropout
            ))
        else:
            self.bottleneck.append(BasicBlock(ch, ch))
```

| 对比项 | Mamba2DLayer | CNNRNNBottleneck |
|--------|-------------|------------------|
| 核心机制 | 四向选择性状态空间扫描 | 双向GRU(行/列) + CNN残差 |
| 参数量 | ~70M (4×独立MambaBlock) | ~8M (2×Bi-GRU + 投影) |
| 反向传播 | 纯Python循环，慢，易OOM | PyTorch原生GRU，CUDA优化 |
| 显存占用 | 大(保存中间状态) | 小(GRU内部优化) |

### 2.2 输出头 (新增 Transformer)

**旧**:
```python
self.final_conv = nn.Conv2d(ch, 1, 1)
# forward: return self.final_conv(x)
```

**新**:
```python
if config.use_transformer_output:
    self.transformer_out = TransformerSpatialBlock(
        ch, config.transformer_num_heads, config.transformer_ffn_ratio, config.transformer_dropout
    )
else:
    self.transformer_out = nn.Identity()
self.final_conv = nn.Conv2d(ch, 1, 1)
# forward: x = self.transformer_out(x); return self.final_conv(x)
```

---

## 3. 配置参数速查

### 3.1 移除的旧参数
```python
# use_mamba: bool = True
# mamba_d_state: int = 16
# mamba_d_conv: int = 4
# mamba_expand: int = 2
```

### 3.2 新增参数
```python
# --- CNN-RNN Bottleneck ---
use_cnn_rnn: bool = True          # 总开关
rnn_hidden_ratio: float = 0.5     # GRU hidden = channels * 0.5
num_gru_layers: int = 1           # Bi-GRU 层数
rnn_dropout: float = 0.0          # GRU dropout

# --- Transformer Output ---
use_transformer_output: bool = True   # 总开关
transformer_num_heads: int = 8        # MSA 头数
transformer_ffn_ratio: int = 4        # FFN hidden = channels * 4
transformer_dropout: float = 0.0      # dropout
```

---

## 4. 性能实测数据

环境: CUDA, BF16, batch=40, board=50×50, RoPE输入

| 指标 | 数值 |
|------|------|
| 总参数量 | ~116.8M |
| 前向+反向峰值显存 | **2.22 GB** |
| 当前占用显存 | 1.78 GB |
| 5GB限制 | ✅ 通过 (余量 ~2.8GB) |

---

## 5. 调参指南

| 场景 | 推荐配置 |
|------|---------|
| **默认** | `use_cnn_rnn=true, use_transformer_output=true` |
| 显存 < 3GB | `use_transformer_output=false, rnn_hidden_ratio=0.25` |
| 大棋盘 (100×100) | `transformer_num_heads=4` (减少注意力计算) |
| 追求速度 | `use_transformer_output=false` |
| 长距离依赖重要 | `num_gru_layers=2, transformer_num_heads=8` |
| 过拟合风险 | `dropout=0.1, rnn_dropout=0.1, transformer_dropout=0.1` |

---

## 6. 关键设计决策

1. **为什么用 GRU 不用 LSTM?**
   - GRU 参数量更少， gates 少一个，训练更快
   - 对于空间扫描任务，GRU 和 LSTM 效果相当

2. **为什么 Row + Col 两个方向?**
   - 三消规则本质上是行/列连续性检测
   - Row-GRU 捕获水平模式，Col-GRU 捕获垂直模式
   - 与 Mamba 四向扫描等价，但更高效

3. **Transformer 放在 Decoder 末端而非 Bottleneck?**
   - Bottleneck 特征图尺寸小(1×1~3×3)，attention 收益低
   - Decoder 末端恢复全分辨率(50×50)，attention 可直接建模全局空间关系
   - 参数量小(~0.5M)，显存开销可控

4. **为什么 channels=32 时 num_heads=8?**
   - 32 / 8 = 4，每个 head 维度为 4
   - 满足 `channels % num_heads == 0` 约束
   - 若改 base_channels，需同步调整 num_heads

---

## 7. 兼容性说明

- ✅ `Match3UNet(config)` 构造器签名不变
- ✅ `forward(x)` 输入输出形状不变
- ✅ `model.cfg` 属性访问兼容 (`getattr` 兜底)
- ❌ **旧检查点不兼容** — 结构变化，需重新训练
- ❌ `Mamba2DLayer` / `MambaBlock` 已移除
