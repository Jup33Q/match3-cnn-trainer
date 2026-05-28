<!--
  SKILL: match3-training-foundation
  DESC: 三消模型训练通用方法论，提供损失设计原则、训练策略、评估标准和调优建议
  AUTHOR: AI Skill System
  VERSION: 1.0.0
  DEPENDS: torch>=2.0.0, tensorboard>=2.13.0
  COMPAT: Python 3.9+
-->

# match3-training-foundation

> 三消模型训练通用方法论
>
> 版本: 1.0.0 | 依赖: torch>=2.0.0, tensorboard>=2.13.0

---

## 1. 核心概念

### 1.1 问题定义

三消Pattern识别本质上是一个**像素级二分类分割问题**：

- **输入**：`board` —— `(H, W)` 的元素类型矩阵，每个位置代表一种游戏元素（如宝石颜色）
- **输出**：`mask` —— `(H, W)` 的二值矩阵，`1` 表示该位置属于某个消除Pattern，`0` 表示不属于

### 1.2 类别分布特征

| 特征 | 说明 | 影响 |
|------|------|------|
| **极端不平衡** | 正样本（Pattern格点）占比通常 < 10% | 标准BCE Loss会偏向负类 |
| **稀疏正样本** | Pattern在棋盘上分布稀疏且局部聚集 | 需要关注空间结构 |
| **边缘敏感** | Pattern边界决定了消除范围 | 边缘精度直接影响游戏体验 |

> **关键洞察**：损失函数设计必须针对类别不平衡问题，不能直接使用标准CrossEntropy。

---

## 2. 损失函数设计原则

### 2.1 Dice Loss

**适用场景**：处理严重的类别不平衡（正样本 < 10%）

**原理**：基于IoU的度量，对正负样本比例不敏感

```python
def dice_loss(pred, target, smooth=1.0):
    """
    pred: (B, H, W) 模型输出（经sigmoid）
    target: (B, H, W) 黄金标准mask
    """
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()
```

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `smooth` | 1.0 | 避免除零，保持数值稳定 |

### 2.2 Focal Loss

**适用场景**：存在大量易分类负样本，需要聚焦难例

**原理**：降低易分类样本的权重，让模型关注难分样本

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    alpha: 正样本权重（平衡正负样本）
    gamma: 聚焦参数（越大越关注难例）
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    fl = alpha * (1 - pt) ** gamma * bce
    return fl.mean()
```

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `alpha` | 0.25 | 正样本权重，用于平衡正负比例 |
| `gamma` | 2.0 | 聚焦强度，gamma越大越忽略易分类样本 |

> **调参建议**：若Recall低，增大alpha；若存在大量噪声误检，增大gamma。

### 2.3 Boundary Loss

**适用场景**：需要强化Pattern边缘定位精度

**原理**：对靠近Pattern边界的预测误差施加更大惩罚

```python
def boundary_loss(pred, target, boundary_weight=5.0):
    """
    对Pattern边界区域的预测误差增加权重
    """
    # 通过形态学操作提取边界区域
    boundary = extract_boundary(target)  # (B, H, W) 二值边界mask
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    weights = 1.0 + boundary_weight * boundary
    return (weights * bce).mean()
```

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `boundary_weight` | 5.0 | 边界区域损失放大倍数 |

### 2.4 组合权重设计建议

**推荐组合方案**：

```
总损失 = λ_dice × DiceLoss + λ_focal × FocalLoss + λ_boundary × BoundaryLoss

推荐权重：
    λ_dice = 0.5    (主力损失，处理不平衡)
    λ_focal = 0.3   (辅助损失，聚焦难例)
    λ_boundary = 0.2 (边缘增强，提升边界精度)
```

> **替代方案**：若训练资源有限，可仅使用 `DiceLoss + FocalLoss`（权重0.7 : 0.3），效果差距通常 < 1% IoU。

---

## 3. 类别不平衡处理策略

### 3.1 NONE类降权

- NONE类（负样本）占比过高（> 90%），在Loss中应适当降权
- 推荐在BCE/Focal Loss中将正类权重设为负类的5~10倍

### 3.2 复杂Pattern升权

若模型支持Pattern类型分类，可对复杂Pattern施加更高权重：

| Pattern类型 | 相对权重 | 原因 |
|-------------|----------|------|
| CROSS | 2.0× | 结构最复杂，样本最少 |
| T | 1.5× | 次复杂结构 |
| L | 1.5× | 含拐角特征 |
| H5/V5 | 1.2× | 长链样本较少 |
| H4/V4 | 1.0× | 基准权重 |
| H3/V3 | 1.0× | 最基础结构，样本充足 |

---

## 4. 训练策略

### 4.1 优化器选择

**推荐**：`AdamW`（Adam with Weight Decay）

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `lr` | 1e-3 ~ 5e-4 | 初始学习率 |
| `weight_decay` | 1e-4 | 权重衰减，防止过拟合 |
| `betas` | (0.9, 0.999) | 默认动量参数 |

### 4.2 学习率调度

**推荐方案对比**：

| 调度器 | 适用场景 | 推荐配置 |
|--------|----------|----------|
| **CosineAnnealing** | 通用首选 | `T_max = total_epochs`, `eta_min = 1e-6` |
| **OneCycleLR** | 快速收敛 | `max_lr = 1e-3`, `pct_start = 0.3` |
| **ReduceLROnPlateau** | 精细调优 | `patience = 5`, `factor = 0.5` |

> **建议**：CosineAnnealing为最稳妥选择，OneCycleLR适合需要快速验证的场景。

### 4.3 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `max_norm` | 1.0 | 梯度L2范数上限，防止梯度爆炸 |

### 4.4 Warmup策略

```
前 warmup_epochs 个epoch线性增加学习率：
    lr = base_lr × (current_step / warmup_steps)

推荐配置：
    warmup_epochs = 3 ～ 5
    warmup_ratio = 0.01 → 1.0（线性）
```

---

## 5. 课程学习规范

### 5.1 棋盘尺寸递进策略

通过逐步增大棋盘尺寸，让模型从简单场景过渡到复杂场景：

| 阶段 | 棋盘尺寸 | Epoch数 | 目的 |
|------|----------|---------|------|
| **Stage 1** | 8 × 8 | 20 | 学习基础Pattern特征 |
| **Stage 2** | 16 × 16 | 30 | 适应中等棋盘，学习空间关系 |
| **Stage 3** | 32 × 32 | 40 | 处理复杂多Pattern场景 |
| **Stage 4** | 64 × 64 | 50+ | 最终目标尺寸，微调精度 |

### 5.2 各阶段配置

```python
class CurriculumStage:
    board_size: Tuple[int, int]   # 当前阶段棋盘尺寸
    num_epochs: int                # 当前阶段训练epoch数
    pattern_density: float         # Pattern密度（随阶段递增）
    loss_weights: dict             # 当前阶段损失权重（可微调）
```

> **设计原理**：小棋盘Pattern结构简单、特征明显，适合早期快速收敛；逐步增大尺寸让模型适应更大范围的空间依赖。

---

## 6. 评估指标体系

### 6.1 核心指标

| 指标 | 公式 | 目标值 |
|------|------|--------|
| **IoU** | `TP / (TP + FP + FN)` | > 0.80（优秀） |
| **Precision** | `TP / (TP + FP)` | > 0.85 |
| **Recall** | `TP / (TP + FN)` | > 0.85 |
| **F1-Score** | `2PR / (P + R)` | > 0.85 |
| **Pixel Accuracy** | `(TP + TN) / Total` | > 0.95（因负样本多，参考价值有限） |

> **关键指标**：**IoU** 是主要评估指标，直接反映分割质量。Pixel Accuracy因类别不平衡参考价值有限。

### 6.2 Pattern分类Accuracy（如适用）

若模型同时输出Pattern类型分类，额外指标：

| 指标 | 说明 |
|------|------|
| **Pattern Acc** | 正样本中Pattern类型预测正确的比例 |
| **Confusion Matrix** | 各类Pattern的混淆情况分析 |

---

## 7. 训练监控规范

### 7.1 TensorBoard指标命名约定

```
# 损失指标
Loss/Total          —— 总损失
Loss/Dice           —— Dice损失分量
Loss/Focal          —— Focal损失分量
Loss/Boundary       —— Boundary损失分量

# 学习率
Train/LR            —— 当前学习率

# 训练指标（每个epoch）
Train/IoU           —— 训练集IoU
Train/Precision     —— 训练集Precision
Train/Recall        —— 训练集Recall
Train/F1            —— 训练集F1

# 验证指标（每个epoch）
Val/IoU             —— 验证集IoU
Val/Precision       —— 验证集Precision
Val/Recall          —— 验证集Recall
Val/F1              —— 验证集F1

# 辅助指标
Train/Grad_Norm     —— 梯度范数
Val/Pattern_Dist    —— Pattern类型分布（histogram）
```

### 7.2 监控建议

- 每 **1 epoch** 记录训练和验证指标
- 每 **100 iterations** 记录损失曲线
- 每 **5 epochs** 保存一次模型checkpoint
- 保留验证IoU最高的模型作为best checkpoint

---

## 8. 调优检查表

### IoU不达标时的排查步骤

```
□ Step 1: 检查数据质量
    └─ 可视化样本board+mask，确认标注正确性
    └─ 检查正样本比例是否在 5% ~ 15% 之间

□ Step 2: 检查损失函数
    └─ Dice Loss是否正常工作（训练初期应快速下降）
    └─ 尝试调整focal_loss的alpha/gamma参数

□ Step 3: 检查学习率
    └─ 学习率是否过大（loss震荡）或过小（loss不下降）
    └─ 尝试ReduceLROnPlateau替代固定调度

□ Step 4: 检查模型容量
    └─ 是否欠拟合（train IoU也低）→ 增大模型
    └─ 是否过拟合（val IoU远低于train）→ 增加正则化

□ Step 5: 检查类别权重
    └─ 正样本权重是否足够高
    └─ 复杂Pattern是否需要额外升权

□ Step 6: 检查数据增强
    └─ 是否使用了合适的空间增强（翻转、旋转）
    └─ 避免破坏Pattern结构的增强

□ Step 7: 检查课程学习
    └─ 是否从足够小的棋盘开始
    └─ 各阶段epoch是否充足
```

---

## 9. 性能预期参考表

### 9.1 不同棋盘尺寸的IoU目标

| 棋盘尺寸 | 随机基线 IoU | 训练后目标 IoU | 优秀 IoU |
|----------|-------------|---------------|---------|
| 8 × 8 | ~0.05 | > 0.85 | > 0.92 |
| 16 × 16 | ~0.03 | > 0.82 | > 0.90 |
| 32 × 32 | ~0.02 | > 0.80 | > 0.88 |
| 64 × 64 | ~0.01 | > 0.78 | > 0.85 |

> **说明**：随机基线指正样本随机猜测的期望IoU，反映了任务难度。

### 9.2 训练时间参考

| 棋盘尺寸 | 数据集大小 | GPU (A100) | GPU (RTX 4090) |
|----------|-----------|------------|----------------|
| 8 × 8 | 10K | ~5 min | ~10 min |
| 16 × 16 | 20K | ~15 min | ~30 min |
| 32 × 32 | 50K | ~1 hr | ~2.5 hr |
| 64 × 64 | 100K | ~4 hr | ~10 hr |

> **注意**：时间为单阶段训练参考值，课程学习总时间为各阶段之和。

---

## 10. 版本兼容性

| 版本 | 兼容范围 | 说明 |
|------|----------|------|
| 1.0.0 | Python 3.9+, PyTorch 2.0+, TensorBoard 2.13+ | 初始版本，支持基础训练方法论 |

---

<!-- SKILL-REF: match3-training-foundation -->
