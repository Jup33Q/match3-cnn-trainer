# Agent Guide: match3-cnn-trainer

> 本文件面向 AI Agent，描述 `match3_cnn_trainer` 项目的核心架构、编码约定和常见修改点。

---

## 1. 项目概述

基于 **CNN-RNN-Transformer U-Net** 的三消 Pattern 标注模型训练器。

- **输入**: 正n边形顶点3通道编码 `(B, 3 + 1, H, W)`，其中 `+1` 是可选的 `valid_mask` 通道
- **输出**: 像素级消除 mask `(B, 1, H, W)`
- **核心任务**: 输入棋盘，输出每个格子是否属于三消 Pattern（H/V/L/T/Cross）

---

## 2. 输入编码（关键！）

### 2.1 正n边形顶点3通道颜色编码

所有 stage 统一使用 **正n边形顶点坐标** 编码颜色：

```python
n = 当前样本颜色种类数  # stage0-2: n=6; stage2.5+: n=5~12
angles = 2 * np.pi * board / n
board_tensor = np.stack([
    np.cos(angles),       # 顶点 x 坐标
    np.sin(angles),       # 顶点 y 坐标
    np.ones_like(angles)  # 常数 1
], axis=0)  # (3, H, W)
```

- **通道数固定为 3**，与颜色种类数 `n` 无关
- 相同颜色的格子在3D空间中欧氏距离为 0
- 不同颜色的距离由正n边形几何关系自然定义
- stage0-2: `n=6` 固定（正六边形顶点）
- stage2.5+: `n` 随机 `5~12`（正n边形顶点）

### 2.2 valid_mask 通道

当 `use_valid_mask=True`（默认）时，第 4 个通道是 `valid_mask`：

```python
valid_mask = torch.ones(1, H, W)  # 1=有效棋盘区域, 0=padding
board_tensor = torch.cat([board_tensor, valid_mask], dim=0)  # (4, H, W)
```

**作用**: stage3.5/4/5 的随机尺寸棋盘被嵌入到 `max_size x max_size` 画布时，`valid_mask` 标记实际棋盘区域，模型据此忽略 padding。

---

## 3. 课程学习（6 阶段递进）

| Stage | 标识 | 尺寸 | 颜色种类 | Match | 输入编码 | 说明 |
|:-----:|:----:|:----:|:--------:|:-----:|:--------:|:-----|
| 0 | `10` | 10×10 | 6 固定 | 3~5 | 正6边形顶点 (3通道) | |
| 1 | `25` | 25×25 | 6 固定 | 3~5 | 正6边形顶点 (3通道) | |
| 2 | `50` | 50×50 | 6 固定 | 3~5 | 正6边形顶点 (3通道) | |
| 2.5 | `-1` | 正方形随机 10~50 | **随机 5~12** | 3~5 | **正n边形顶点 (3通道)** | 尺寸+颜色同时变化 |
| 3 | `-3` | 长方形随机 10~50 | **随机 5~12** | 3~5 | **正n边形顶点 (3通道)** | 宽高独立随机 |
| 4 | `-2` | 长方形随机 10~50 | **随机 5~12** | 5~8 | **正n边形顶点 (3通道)** | match长度也随机 |

### 3.1 Stage 标识约定

- **正数**: 固定正方形尺寸，`n=6` 固定
- **`-1`**: 正方形随机过渡阶段，`n` 随机 5~12
- **`-3`**: 长方形随机阶段（宽高独立随机 10~50），`n` 随机 5~12
- **`-2`**: stage5 全面泛化（长方形随机 + `n` 随机 5~12 + match 长度随机 5~8）

### 3.2 长方形随机实现

`generate_board(height, width, ...)` 支持非正方形棋盘：

```python
actual_h = random.randint(10, max_size)
actual_w = random.randint(10, max_size)
board, mask, _ = generator.generate_board(actual_h, actual_w, difficulty)
```

嵌入大画布时，长方形棋盘被放置在 `max_size x max_size` 画布的随机位置：

```python
sy = random.randint(0, max_size - actual_h)
sx = random.randint(0, max_size - actual_w)
full_board[sy:sy+actual_h, sx:sx+actual_w] = board
```

---

## 4. 模型架构

```
Input: (B, 4, H, W)  [3 正n边形顶点编码 + 1 valid_mask]
  ↓
[Stem: 7×7 Conv] ───────────────────────────→ 32ch
  ↓
[Encoder Stage 1~5] 2×ResBlock + Pool ──────→ 1024ch, 1×1
  ↓
[Bottleneck] Soft-Router + 并行CNN分支 ─────→ 1024ch
  ↓
[Decoder Stage 5~1] Up + 2×ResBlock ────────→ 32ch, 50×50
  ↓
[Transformer Block] MSA + FFN ──────────────→ 32ch
  ↓
[Output] 1×1 Conv ──────────────────────────→ 1ch, logits
```

- **Stem**: `input_ch = 3 + (1 if use_valid_mask else 0) = 4`
- **Bottleneck**: Soft-Router 根据输入特征动态加权多个并行 CNN 分支
- **Transformer Output**: Decoder 末端可选 Pre-LN MSA + FFN

---

## 5. 关键文件说明

| 文件 | 职责 | Agent 修改注意点 |
|------|------|-----------------|
| `config.py` | 全局配置 | 新增配置项需在此定义默认值 |
| `data/data_generator.py` | 棋盘生成 + 正n边形编码 | `generate_board()` 已支持 `(height, width)` 长方形；`_generate_one()` 中编码逻辑是核心 |
| `models/model.py` | U-Net 模型定义 | `stem` 的 `input_ch` 固定为 `3 + valid_mask` |
| `trainer/trainer.py` | 训练循环 + 课程学习 | `fit()` 中 `stage_label` 需为新 stage 标识添加分支 |
| `trainer/inference.py` | 推理封装 | `_encode_board()` 使用正n边形编码，与训练一致 |
| `models/postprocess.py` | 三消规则后处理 | `enforce_match3_rules()` 用 BFS + 直线连续性验证 |

---

## 6. 编码约定

### 6.1 新增 Stage 标识的流程

若需新增课程学习阶段（如 `-4`）：

1. **`config.py`**: 更新 `curriculum_stages` 默认值
2. **`data_generator.py`**:
   - `Match3Dataset.__init__`: 添加 `self.new_stage = (stage_size == -4)`
   - `preload` 条件排除新 stage（若需动态生成）
   - `_generate_one`: 添加新 stage 的参数逻辑（尺寸、颜色种类 n）
3. **`trainer/trainer.py`**: `fit()` 中 `stage_label` 添加新分支
4. **`README.md + AGENTS.md`**: 更新课程学习表格

### 6.2 颜色编码修改

正n边形顶点编码集中在 `data_generator.py` 和 `inference.py` 的 `_encode_board` 中：

```python
n = num_fruit_types if num_fruit_types is not None else self.cfg.num_fruit_types
angles = 2 * np.pi * board / n
board_tensor = np.stack([np.cos(angles), np.sin(angles), np.ones_like(angles)], axis=0)
```

若需改为其他编码（如 RGB、HSV 等），只需修改这两处。

### 6.3 清除历史并重新训练

当编码方案、模型架构或课程学习阶段发生变更时，旧 checkpoint 可能不兼容：

```bash
# 1. 清除所有训练历史
rm -rf checkpoints/ logs/

# 2. 运行测试确认代码无误
python tests/test_model_new.py
python tests/test_training.py

# 3. 从零开始完整 6 阶段训练
python main.py --mode train --config configs/unet_50x50.json
```

**必须重新训练的场景**:
- 输入编码方案改变（如 RoPE/One-Hot → 正n边形顶点编码）
- `curriculum_stages` 增删阶段
- 模型输入通道数变化且语义不同

### 6.4 检查点兼容性

- **通道数变化**: `load_checkpoint()` 会自动对 `stem.0.weight` 做零填充适配
- **语义变化**: 编码方式的根本改变（如正n边形 → RGB）**不会**自动适配，需重新训练

---

## 7. 测试验证

| 测试文件 | 验证内容 | 运行方式 |
|---------|---------|---------|
| `tests/test_model_new.py` | U-Net 架构前向/梯度/CUDA | `python tests/test_model_new.py` |
| `tests/test_training.py` | 端到端训练 + 推理 + 评估 | `python tests/test_training.py` |
| `tests/test_pattern_core.py` | PatternDetector 逻辑 | `python tests/test_pattern_core.py` |

> **Agent 修改代码后，必须运行 `test_model_new.py` 和 `test_training.py` 验证无报错。**

---

## 8. 快捷工作流（常用脚本）

> 当用户说"帮我push"时，按以下顺序执行：

### 8.1 导出 TensorBoard 训练数据

```bash
python scripts/export_tensorboard.py
```

输出：
- `logs/training_loss.json` — 结构化 scalar 数据
- `logs/TRAINING_LOG.md` — 人类可读 Markdown 报告（损失用科学计数法）

### 8.2 绘制训练曲线（处理跨 stage 数量级差异）

```bash
python scripts/plot_training_curves.py
```

输出（`logs/training_curves/`）：
- `01_global_step_overview.png` — 全局 Step 级 Total Loss（log y + stage 背景色带）
- `02_stage{N}_train_loss.png` — 各 stage 训练损失分解（log y，total/dice/focal/boundary）
- `03_validation_metrics.png` — 验证指标四宫格（iou/f1/precision/recall，线性 y）
- `04_step_level_detail.png` — Step 级损失分面图（log y）
- `05_learning_rate.png` — 学习率变化
- `06_combined_epoch_loss.png` — 合并 epoch 级总损失 + 低损失区域 zoom inset

> **数量级差异处理**：训练损失用 `log scale`（跨度 1e-8 ~ 1e-2），验证指标用线性坐标（0~1）。

### 8.3 Git 提交并 Push

```bash
# 确保 .gitignore 允许日志产物
git add .gitignore scripts/export_tensorboard.py scripts/plot_training_curves.py \
    logs/training_loss.json logs/TRAINING_LOG.md logs/training_curves/

git commit -m "docs: update training logs and curves"
git push origin main
```

> **注意**：只 add 与日志/可视化相关的文件，不提交训练中间产物（checkpoints/大量 visualization PNG）。
