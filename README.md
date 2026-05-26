# Match-3 CNN U-Net Pattern Recognition Trainer (50×50)

> 基于 **深度 ResNet-U-Net** 的三消（Match-3）Pattern 标注模型训练器。
>
> 支持 50×50 棋盘、BF16 混合精度训练、课程学习、显存监控、断点恢复。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.9 |
| CUDA | >= 13.0 (推荐) |
| GPU 显存 | >= 8GB (训练峰值 ~7GB，batch=200) |

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
│   ├── unet_50x50.json          # 50×50 默认配置 (batch=200)
│   └── unet_50x50_resume.json   # 恢复训练配置 (batch=80, 更稳定)
├── scripts/                     # 训练快捷脚本
│   ├── resume_from_epoch50.sh   # 从 Epoch 50 恢复
│   ├── resume_reduce_memory.sh  # 低显存模式恢复
│   └── train_from_scratch.sh    # 从头训练
├── models/                      # 模型模块
│   ├── __init__.py
│   ├── model.py                 # 深度 ResNet-U-Net 模型
│   ├── mamba_layer.py           # 可选 Mamba2D 状态空间层
│   └── postprocess.py           # 三消规则后处理
├── trainer/                     # 训练模块
│   ├── __init__.py
│   ├── trainer.py               # BF16 训练引擎 + 课程学习
│   ├── losses.py                # Dice + Focal + Boundary 组合损失
│   ├── inference.py             # 推理与评估接口
│   └── memory_monitor.py        # GPU/系统内存监控
├── data/                        # 数据模块
│   ├── __init__.py
│   ├── data_generator.py        # 程序化棋盘与 Mask 生成
│   └── pattern_types.py         # PatternType 枚举 + PatternDetector
├── utils/                       # 工具脚本
│   └── model_summary.py         # 模型结构与参数计算器
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
python utils/model_summary.py --batch_size 200 --board_size 50
```

### 3. 运行测试验证

```bash
# Pattern 核心逻辑测试
python tests/test_pattern_core.py

# 显存占用实测 (batch=200, 50×50)
python tests/test_memory.py

# 端到端迷你训练测试
python tests/test_training.py
```

### 4. 正式训练

```bash
# 使用默认配置从头训练
python main.py --mode train --config configs/unet_50x50.json
```

#### 快捷脚本

```bash
# 从头开始训练
bash scripts/train_from_scratch.sh
```

### 5. 从检查点恢复训练

训练中断或想基于已有模型继续训练时：

```bash
# 从最佳模型恢复
python main.py --mode train \
    --checkpoint ./checkpoints/match3_unet_epochbest.pt \
    --resume

# 从第 50 轮恢复 (进入 50×50 阶段，batch=80)
python main.py --mode train \
    --config configs/unet_50x50_resume.json \
    --checkpoint ./checkpoints/match3_unet_epoch50.pt \
    --resume \
    --start_stage 2
```

> `--resume` 会加载模型权重、优化器状态、学习率调度器状态以及最佳 IoU，
> 自动跳过已完成的 epoch，从断点继续训练。

#### 快捷脚本

```bash
# 从 Epoch 50 恢复训练 (课程阶段 2: 50×50, batch=80)
bash scripts/resume_from_epoch50.sh

# 低显存模式恢复 (启用 CUDA 显存优化，适用于接近 OOM 的情况)
bash scripts/resume_reduce_memory.sh
```

### 6. TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs --port=6006

# 查看指标
#   Scalars: train_step/total, val/iou, val/f1, memory/gpu_*
#   Images: confusion/pattern (每 5 epoch)
```

### 7. 模型评估

```bash
python main.py --mode eval \
    --checkpoint ./checkpoints/match3_unet_epochbest.pt
```

### 8. 单张推理

```bash
# 准备棋盘 .npy 文件 (shape: 50×50, dtype=int64)
python main.py --mode infer \
    --checkpoint ./checkpoints/match3_unet_epochbest.pt \
    --board data/test_board.npy

# 输出: predicted_mask.npy
```

---

## 配置说明

核心配置参数 (`config.py` / `configs/*.json`)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `board_size` | 50 | 棋盘尺寸 (50×50) |
| `batch_size` | 200 | 每 batch 样本数 |
| `precision` | "bf16" | 训练精度: bf16 / fp16 / fp32 |
| `base_channels` | 32 | 首层通道数 |
| `num_encoder_blocks` | 5 | Encoder/Decoder 层数 |
| `blocks_per_stage` | 3 | 每 Stage ResBlock 数量 |
| `bottleneck_blocks` | 4 | Bottleneck ResBlock 数量 |
| `use_mamba` | false | 是否在 Bottleneck 中启用 Mamba2D 层 |
| `mamba_d_state` | 16 | Mamba 状态空间维度 |
| `mamba_expand` | 2 | Mamba 内部扩展因子 |
| `use_dilation` | true | 是否启用膨胀卷积 |
| `dilation_rates` | [1,2,4,8] | 各层膨胀率 |
| `curriculum_enabled` | true | 是否启用课程学习 |
| `curriculum_stages` | [10,25,50] | 棋盘尺寸递进阶段 |
| `save_every_batches` | 10 | 每 N batches 记录日志 |
| `save_every_epochs` | 10 | 每 N epochs 保存检查点 |
| `visualize_every_epochs` | 5 | 每 N epochs 生成可视化样例 (0=关闭) |
| `max_memory_gb` | 8.0 | 显存告警阈值 (GB) |
| `pin_memory` | true | 是否启用 DataLoader 的 pin_memory |
| `prefetch_factor` | 4 | 每个 worker 预取 batch 数 |

### 显存与稳定性调节指南

当前默认配置 (`base=32, 5层, stage_blocks=3, bot_blocks=4, batch=200`)：
- **参数量**: ~174M (启用 Mamba 后 ~204M)
- **BF16 训练显存**: ~7 GB (启用 Mamba 后 ~9-10 GB)

#### 降低显存 / 提升稳定性

```python
# 减小 batch (显存下降最明显)
batch_size = 80   # ~4-5GB

# 减少 DataLoader worker (减少 CPU 负载和内存占用)
num_workers = 2

# 降低模型容量
base_channels = 24
blocks_per_stage = 2
bottleneck_blocks = 3

# 更轻量 (~2.0GB / 155M 参数)
blocks_per_stage = 3
bottleneck_blocks = 3

# 关闭可视化 (减少 CPU 阻塞导致的 GPU 空闲)
visualize_every_epochs = 0
```

#### 训练不稳定 (GPU 利用率锯齿波)

如果出现 GPU 利用率剧烈波动：
1. 检查 `visualize_every_epochs` 是否 > 0，适当增大（默认 5）
2. 确保 `persistent_workers` 生效（需要 `num_workers > 0`）
3. 降低 `num_workers` 如果 CPU 成为瓶颈
4. 使用 `resume_reduce_memory.sh` 启用 CUDA 显存分配优化

---

## 模型架构

**深度 ResNet-U-Net**

```
Input (50×50)
  ↓
[Stem: 7×7 Conv] ───────────────────────────→ 32ch
  ↓
[Encoder Stage 1] 3×ResBlock + Pool ────────→ 64ch,  25×25
[Encoder Stage 2] 3×ResBlock + Pool ────────→ 128ch, 12×12
[Encoder Stage 3] 3×ResBlock + Pool ────────→ 256ch, 6×6
[Encoder Stage 4] 3×ResBlock + Pool ────────→ 512ch, 3×3
[Encoder Stage 5] 3×ResBlock + Pool ────────→ 1024ch, 1×1
  ↓
[Bottleneck] 4×ResBlock ────────────────────→ 1024ch, 1×1
  ↓
[Decoder Stage 5~1] Up + 3×ResBlock ────────→ 32ch, 50×50
  ↓
[Output] 1×1 Conv ──────────────────────────→ 1ch, logits
```

**架构亮点**：
- **ResNet 残差连接**: 每个 BasicBlock 含 2×(3×3) + Shortcut，缓解梯度消失
- **膨胀卷积**: Encoder 深层使用 dilation=[1,2,4,8] 扩大感受野
- **Logits 输出**: 模型输出 logits，损失函数内部分别做 sigmoid / bce_with_logits，兼容 BF16 autocast
- **后处理规则层**: BFS 连通分量 + 直线连续性验证，确保只识别合法三消 Pattern

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
| `test_memory.py` | batch=200 显存实测 | ✅ ~7GB |
| `test_training.py` | 端到端训练+推理+评估 | ✅ 通过 |

---

## 许可证

本项目为内部 Skill & Agent 实现，基于 `skills_and_agent/` 目录中的规范构建。
