# Match-3 CNN U-Net Pattern Recognition Trainer (50×50)

> 基于 **深度 ResNet-U-Net** 的三消（Match-3）Pattern 标注模型训练器。
>
> 支持 50×50 棋盘、BF16 混合精度训练、课程学习（5阶段递进）、RoPE Fruit 编码、显存监控、断点恢复。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.9 |
| CUDA | >= 13.0 (推荐) |
| GPU 显存 | >= 8GB (训练峰值 ~7GB，batch=80) |

## 依赖安装

### ⚠️ 前提说明

**PyTorch 必须按 CUDA 版本单独安装**，不能直接通过 `requirements.txt` 安装（否则会装到 CPU 版本）。

### 方式一：从零开始（推荐）

```bash
# 1. 创建并激活 conda 环境
conda create -n match3 python=3.12 -y
conda activate match3

# 2. 先安装 pip
conda install pip -y

# 3. 按 CUDA 版本安装 PyTorch + torchvision
#    CUDA 13.2 用户:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

#    其他 CUDA 版本请去 https://pytorch.org/get-started/locally/ 查找对应命令

# 4. 安装其余依赖
pip install -r requirements.txt
```

### 方式二：已有 PyTorch 环境

如果你已经装好了对应 CUDA 版本的 PyTorch，只需：

```bash
pip install -r requirements.txt
```

### 可选：启用 Mamba 层

如需在模型中使用 Mamba 状态空间模型层：

```bash
# 必须先装好 torch，再安装 mamba-ssm
pip install mamba-ssm causal-conv1d
# 或参考官方指引: https://github.com/state-spaces/mamba
```

---

## 项目结构

```
match3_cnn_trainer/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── main.py                      # CLI 入口 (train/eval/infer)
├── config.py                    # 参数化配置管理
├── configs/
│   ├── unet_50x50.json              # 50×50 默认配置 (batch=40)
│   ├── unet_50x50_resume.json       # 恢复训练配置 (batch=40)
│   └── unet_50x50_resume_epoch40.json  # 恢复训练 + Dropout 配置
├── scripts/                     # 训练快捷脚本
│   ├── resume_from_epoch50.sh   # 从最佳检查点恢复 (Stage 2)
│   ├── resume_reduce_memory.sh  # 低显存模式恢复
│   ├── resume_with_dropout.sh   # 恢复训练并启用 Dropout
│   └── train_from_scratch.sh    # 从头训练
├── models/                      # 模型模块
│   ├── __init__.py
│   ├── model.py                 # 深度 ResNet-U-Net 模型 (RoPE 输入)
│   ├── mamba_layer.py           # 可选 Mamba2D 状态空间层
│   └── postprocess.py           # 三消规则后处理 (支持 fruit_ids)
├── trainer/                     # 训练模块
│   ├── __init__.py
│   ├── trainer.py               # BF16 训练引擎 + 5阶段课程学习
│   ├── losses.py                # Dice + Focal + Boundary 组合损失
│   ├── inference.py             # 推理与评估接口
│   └── memory_monitor.py        # GPU/系统内存监控
├── data/                        # 数据模块
│   ├── __init__.py
│   ├── data_generator.py        # 程序化棋盘与 Mask 生成 (含 FruitRoPE)
│   └── pattern_types.py         # PatternType 枚举 + PatternDetector
├── utils/                       # 工具脚本
│   ├── model_summary.py         # 模型结构与参数计算器
│   ├── draw_architecture.py     # Matplotlib 架构图生成
│   ├── draw_architecture_pytorch.py  # PyTorch 原生可视化 (torchinfo/TensorBoard/ONNX)
│   └── draw_architecture_torchviz.py # torchviz 计算图可视化
└── tests/                       # 测试脚本
    ├── test_pattern_core.py     # Pattern 核心逻辑测试
    ├── test_memory.py           # 显存占用实测
    └── test_training.py         # 端到端训练流程验证
```

---

## 快速开始

### 1. 激活环境

```bash
source /home/jup33q/miniconda3/bin/activate mcp
```

### 2. 查看模型结构与显存估算

```bash
cd match3_cnn_trainer
python utils/model_summary.py --batch_size 40 --board_size 50
```

### 3. 运行测试验证

```bash
# Pattern 核心逻辑测试
python tests/test_pattern_core.py

# 显存占用实测 (batch=80, 50×50)
python tests/test_memory.py

# 端到端迷你训练测试
python tests/test_training.py
```

### 4. 训练脚本使用 (`main.py`)

`main.py` 是统一的 CLI 入口，支持 `train`（训练）、`eval`（评估）、`infer`（推理）三种模式。

#### 基础命令

```bash
# 查看帮助
python main.py --help

# 使用默认配置从头训练
python main.py --mode train --config configs/unet_50x50.json
```

#### 常用参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode` | 运行模式：`train` / `eval` / `infer` | `--mode train` |
| `--config` | JSON 配置文件路径 | `--config configs/unet_50x50.json` |
| `--checkpoint` | 模型检查点 `.pt` 文件路径 | `--checkpoint checkpoints/match3_unet_epochbest.pt` |
| `--resume` | 从 `--checkpoint` 恢复训练（加载优化器、学习率、epoch 状态） | `--resume` |
| `--start_stage` | 课程学习从第几阶段开始（0-based） | `--start_stage 2` |
| `--board` | 推理模式用的棋盘 `.npy` 文件（仅 infer） | `--board data/test_board.npy` |

#### 基于现有 Checkpoint 继续训练

如果你已经有一个 `.pt` 检查点（如 `match3_unet_epochbest.pt` 或 `match3_unet_epochfinal.pt`），有两种使用方式：

**A. 恢复训练（继续之前的训练状态）**
使用 `--resume` 会同时加载：
- 模型权重（兼容旧 one-hot 输入 → 新 RoPE 输入，自动零填充适配）
- 优化器状态
- 学习率调度器状态
- 最佳 IoU 记录
- 已训练 epoch 数（自动跳过已完成的 epoch）

```bash
# 从最佳模型恢复训练
python main.py --mode train \
    --checkpoint ./checkpoints/match3_unet_epochbest.pt \
    --resume

# 从最终模型恢复，并从课程阶段 2 开始 (50×50)
python main.py --mode train \
    --config configs/unet_50x50_resume.json \
    --checkpoint ./checkpoints/match3_unet_epochfinal.pt \
    --resume \
    --start_stage 2
```

**B. 基于已有权重开启新训练（不恢复优化器状态）**
如果你只想加载模型权重作为预训练，但不想恢复优化器和 epoch 进度，**不要加 `--resume`**。此时需要在 `main.py` 中手动加载权重：

```python
# 在 main.py 的 train 分支中手动加载权重示例
trainer = Match3Trainer(config)
if args.checkpoint:
    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INIT] 已加载预训练权重: {args.checkpoint}")
trainer.fit()
```

> 💡 提示：`--resume` 必须与 `--checkpoint` 同时使用，否则 `--checkpoint` 在训练模式下不会被加载。

#### 快捷脚本

项目提供了几个常用脚本，方便一键执行：

```bash
# 方式1: 使用快捷脚本 (推荐)
bash scripts/train_from_scratch.sh

# 方式2: 直接运行 main.py (从零开始，课程学习 stage 0)
python main.py --mode train --config configs/unet_50x50.json

# 从零开始，且强制从课程阶段 0 开始 (不加载任何检查点)
python main.py --mode train --config configs/unet_50x50.json --start_stage 0
```

> 💡 **从 0 开始训练的关键**: 不指定 `--checkpoint` 和 `--resume`，模型将随机初始化，课程学习从 Stage 0 (10×10) 开始递进。

#### 继续已有 Checkpoint 训练

**A. 恢复训练（继续之前的训练状态）**
使用 `--resume` 会同时加载模型权重、优化器状态、学习率调度器和已训练 epoch 数：

```bash
# 从最佳模型恢复训练
python main.py --mode train \
    --checkpoint ./checkpoints/match3_unet_epochbest.pt \
    --resume

# 从最终模型恢复，并从课程阶段 2 开始 (50×50)
python main.py --mode train \
    --config configs/unet_50x50_resume.json \
    --checkpoint ./checkpoints/match3_unet_epochfinal.pt \
    --resume \
    --start_stage 2
```

**B. 基于已有权重开启新训练（不恢复优化器状态）**
如果你只想加载模型权重作为预训练，但不想恢复优化器和 epoch 进度，**不要加 `--resume`**：

```python
# 在 main.py 的 train 分支中手动加载权重示例
trainer = Match3Trainer(config)
if args.checkpoint:
    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INIT] 已加载预训练权重: {args.checkpoint}")
trainer.fit()
```

> 💡 提示：`--resume` 必须与 `--checkpoint` 同时使用，否则 `--checkpoint` 在训练模式下不会被加载。

#### 更多快捷脚本

```bash
# 从最佳检查点恢复训练 (课程阶段 2: 50×50)
bash scripts/resume_from_epoch50.sh

# 低显存模式恢复 (启用 CUDA 显存分配优化，适用于接近 OOM 的情况)
bash scripts/resume_reduce_memory.sh

# 恢复训练并启用 Dropout (默认从 Stage 1 开始，可传参指定阶段)
bash scripts/resume_with_dropout.sh    # Stage 1 (25×25)
bash scripts/resume_with_dropout.sh 2  # Stage 2 (50×50)
```

### 5. TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs --port=6006

# 查看指标
#   Scalars: train_step/total, val/iou, val/f1, memory/gpu_*
#   Images: visualizations/epoch*.png (每个 epoch 自动合并输出)
```

### 6. 模型评估

```bash
python main.py --mode eval \
    --checkpoint ./checkpoints/match3_unet_epochbest.pt
```

### 7. 单张推理

```bash
# 准备棋盘 .npy 文件 (shape: 50×50, dtype=int64)
python main.py --mode infer \
    --checkpoint ./checkpoints/match3_unet_epochbest.pt \
    --board data/test_board.npy

# 输出: predicted_mask.npy
```

---

## 检查点管理

训练过程中检查点会自动保存到 `./checkpoints/` 目录：

| 文件名 | 说明 |
|--------|------|
| `match3_unet_epochbest.pt` | 验证 IoU 最高时的最佳模型 |
| `match3_unet_epoch{N}.pt` | 每 N 个 epoch 的常规检查点（若存在） |
| `match3_unet_epochfinal.pt` | 训练结束时的最终模型 |

> 检查点使用 BF16 压缩存储，体积约为 FP32 的 50%。

### 旧检查点兼容

从 one-hot 输入（6 通道）旧检查点迁移到 RoPE 输入（32 通道）时，训练器会自动对 `stem.0.weight` 进行**零填充适配**：
- 旧权重的前 6 个通道直接复制
- 新增的 26 个通道初始化为零
- 其余层权重完全兼容

> 建议在 stage5 开始前（epoch 81）确认模型已加载并适配成功。

---

## Git 提交（可选）

项目支持手动将检查点提交到 Git LFS。保存 `best` / `final` 检查点时会自动执行 `git commit`，但**不会自动 push**。

如需手动推送到远程仓库：

```bash
# GitHub
git push origin main

# ModelScope（国内速度快）
git push modelscope main
```

首次推送可能需要配置 Git LFS，详见 `scripts/setup_git_push.sh`。

---

## 配置说明

核心配置参数 (`config.py` / `configs/*.json`)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `board_size` | 50 | 棋盘尺寸 (50×50)，最大嵌入尺寸 |
| `batch_size` | 40 | 每 batch 样本数 |
| `precision` | "bf16" | 训练精度: bf16 / fp16 / fp32 |
| `base_channels` | 32 | 首层通道数 |
| `num_encoder_blocks` | 5 | Encoder/Decoder 层数 |
| `blocks_per_stage` | 2 | 每 Stage ResBlock 数量 |
| `bottleneck_blocks` | 2 | Bottleneck ResBlock 数量 |
| `use_mamba` | **true** | 是否在 Bottleneck 中启用 Mamba2D 层 (默认启用) |
| `mamba_d_state` | 16 | Mamba 状态空间维度 |
| `mamba_expand` | 2 | Mamba 内部扩展因子 |
| `use_dilation` | true | 是否启用膨胀卷积 |
| `dilation_rates` | [1,2,4,8] | 各层膨胀率 |
| `curriculum_enabled` | true | 是否启用课程学习 |
| `curriculum_stages` | [10,25,50,-1,-2] | 棋盘尺寸递进阶段（JSON 默认配置为 [10,25,50]） |
| `curriculum_epochs_per_stage` | 20 | 每阶段训练 epoch 数 |
| `fruit_embed_dim` | 32 | Fruit RoPE 编码维度 |
| `max_fruit_types` | 16 | RoPE 支持的最大 fruit 种类数 |
| `stage5_match_length_range` | (5,8) | Stage5 match 长度随机范围 |
| `stage5_fruit_range` | (5,12) | Stage5 fruit 种类随机范围 |
| `save_every_batches` | 10 | 每 N batches 记录日志 |
| `save_every_epochs` | 10 | 每 N epochs 保存检查点 |
| `visualize_every_epochs` | 5 | (已弃用) 当前版本每个 epoch 自动生成可视化样例并合并输出 |
| `max_memory_gb` | 8.0 | 显存告警阈值 (GB) |
| `pin_memory` | true | 是否启用 DataLoader 的 pin_memory |
| `prefetch_factor` | 4 | 每个 worker 预取 batch 数 |

### 显存与稳定性调节指南

当前默认配置 (`base=32, 5层, stage_blocks=2, bot_blocks=2, batch=40, use_mamba=true`)：
- **参数量**: ~120M (关闭 Mamba 后 ~90M)
- **BF16 训练显存**: ~4–5 GB (关闭 Mamba 后 ~3–4 GB)

#### 降低显存 / 提升稳定性

```python
# 减小 batch (显存下降最明显)
batch_size = 20   # ~3-4GB

# 减少 DataLoader worker (减少 CPU 负载和内存占用)
num_workers = 2

# 降低模型容量
base_channels = 24
blocks_per_stage = 2
bottleneck_blocks = 2

# 更轻量 (~2.0GB / 70M 参数)
base_channels = 24
blocks_per_stage = 2
bottleneck_blocks = 2

# 如需减少 CPU 阻塞，可降低 num_workers 或关闭其他日志
# num_workers = 0
```

#### 训练不稳定 (GPU 利用率锯齿波)

如果出现 GPU 利用率剧烈波动：
1. 降低 `num_workers` 以减少 CPU 数据加载开销
2. 确保 `persistent_workers` 生效（需要 `num_workers > 0`）
3. 降低 `num_workers` 如果 CPU 成为瓶颈
4. 使用 `resume_reduce_memory.sh` 启用 CUDA 显存分配优化

---

## 模型架构

![Architecture Diagram](./architecture_diagram.png)

**深度 ResNet-U-Net (RoPE 输入版)**

```
Input: Fruit RoPE Embedding (B, 32, H, W)
  ↓
[Stem: 7×7 Conv] ───────────────────────────→ 32ch
  ↓
[Encoder Stage 1] 2×ResBlock + Pool ────────→ 64ch,  25×25
[Encoder Stage 2] 2×ResBlock + Pool ────────→ 128ch, 12×12
[Encoder Stage 3] 2×ResBlock + Pool ────────→ 256ch, 6×6
[Encoder Stage 4] 2×ResBlock + Pool ────────→ 512ch, 3×3
[Encoder Stage 5] 2×ResBlock + Pool ────────→ 1024ch, 1×1
  ↓
[Bottleneck] ResBlock + Mamba2DLayer ───────→ 1024ch, 1×1
  ↓
[Decoder Stage 5~1] Up + 2×ResBlock ────────→ 32ch, 50×50
  ↓
[Output] 1×1 Conv ──────────────────────────→ 1ch, logits
```

**架构亮点**：
- **Fruit RoPE 编码**: 使用 1D Rotary Position Embedding 将 fruit type ID 编码为 32 维向量，替代传统 one-hot，天然支持 5~12 种动态 fruit 种类
- **ResNet 残差连接**: 每个 BasicBlock 含 2×(3×3) + Shortcut，缓解梯度消失
- **膨胀卷积**: Encoder 深层使用 dilation=[1,2,4,8] 扩大感受野
- **Logits 输出**: 模型输出 logits，损失函数内部分别做 sigmoid / bce_with_logits，兼容 BF16 autocast
- **后处理规则层**: BFS 连通分量 + 直线连续性验证，确保只识别合法三消 Pattern

### Fruit RoPE 编码原理

```python
# 1D RoPE: 将 fruit type ID (0~15) 编码为固定 32 维向量
for i in range(0, 32, 2):
    theta = 10000 ^ (-2*i/32)
    emb[i]   = sin(t * theta)
    emb[i+1] = cos(t * theta)
```

- 优点：与 fruit 种类数无关，同一编码器支持 5~12 种动态变化
- 训练时：所有 stage 统一使用 RoPE，旧 one-hot checkpoint 可零填充适配

---

## 课程学习 (5 阶段递进)

| 阶段 | 标识 | 棋盘尺寸 | Fruit 种类 | Match 长度 | 输入编码 | Epoch 范围 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Stage 1 | `10` | 10×10 | 6 (固定) | 3~5 (固定) | RoPE | 1–20 |
| Stage 2 | `25` | 25×25 | 6 (固定) | 3~5 (固定) | RoPE | 21–40 |
| Stage 3 | `50` | 50×50 | 6 (固定) | 3~5 (固定) | RoPE | 41–60 |
| Stage 4 | `-1` | **随机 10~50** | 6 (固定) | 3~5 (固定) | RoPE | **61–80** |
| **Stage 5** | `-2` | **随机 10~50** | **随机 5~12** | **随机 5~8** | **RoPE** | **81–100** |

**Stage 4**: 棋盘尺寸随机化泛化训练，每样本独立随机选取 10~50 的边长，嵌入到 50×50 画布中。

**Stage 5**: 全面泛化训练：
- 棋盘尺寸：每样本随机 10~50
- Fruit 种类：每样本随机 5~12 种（通过 RoPE 统一编码）
- Match 长度：每样本随机 5~8（训练长连消除识别）
- 目的是让模型泛化到任意尺寸、任意 fruit 种类数、任意消除长度的场景

> 💡 **注意**: 默认 JSON 配置 (`unet_50x50.json`) 中 `curriculum_stages` 设置为 `[10, 25, 50]`（前 3 阶段），如需完整 5 阶段训练，请修改为 `[10, 25, 50, -1, -2]` 或直接使用 `config.py` 中的默认值。

---

## Pattern 类型支持

| Pattern | 说明 | 最小格点数 |
|---------|------|-----------|
| NONE | 无消除 | — |
| H3 / H4 / H5 | 横向 3/4/5 连 | 3/4/5 |
| V3 / V4 / V5 | 纵向 3/4/5 连 | 3/4/5 |
| L | L型 (拐角) | 5 |
| T | T型 (丁字) | 5 |
| CROSS | 十字型 | 5 |

**优先级**: CROSS > T > L > H5/V5 > H4/V4 > H3/V3 > NONE

---

## 测试验证

| 测试脚本 | 说明 | 状态 |
|----------|------|------|
| `test_pattern_core.py` | PatternDetector 逻辑验证 (H/V/L/T/Cross) | ✅ 8/8 |
| `test_memory.py` | batch=80 显存实测 | ✅ ~5-6GB |
| `test_training.py` | 端到端训练+推理+评估 | ✅ 通过 |

---

## 许可证

本项目为内部 Skill & Agent 实现，基于 `skills_and_agent/` 目录中的规范构建。
