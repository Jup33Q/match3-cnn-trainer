<!--
  Skill: match3-shared-utils
  Description: 三消模型训练共享工具集，提供显存监控、推理API规范、CLI工具和检查点管理
  Version: 1.0.0
  Author: AI Skill Designer
  Stack: PyTorch >=2.0.0, Python >=3.9
  Dependencies: torch>=2.0.0, psutil>=5.9.0, tensorboard>=2.13.0
  Base Skills: match3-training-foundation
-->

# match3-shared-utils

## 1. 概述

本Skill定义三消（Match-3）模型训练 pipeline 的共享工具集规范，包括：

- **显存监控器**: GPU/系统内存实时监控与告警
- **推理API**: 统一预测接口，支持单张/批量推理
- **CLI入口**: 命令行工具规范，支持train/eval/infer三种模式
- **检查点管理**: 模型保存/加载/自动保存策略

> **设计原则**: 与具体模型架构解耦，通过配置文件驱动，可被 `match3-cnn-unet-trainer` 和 `match3-hybrid-transformer-trainer` 复用。

---

## 2. 显存监控器规范

### 2.1 接口设计

```python
class MemoryMonitor:
    """GPU显存与系统内存监控器"""

    def __init__(self, warning_threshold=0.85, log_interval=10):
        self.warning_threshold = warning_threshold  # 显存告警阈值
        self.log_interval = log_interval            # 日志记录间隔(秒)
        self.peak_gpu_mb = 0                        # GPU峰值(MB)
        self.peak_ram_mb = 0                        # RAM峰值(MB)
        self.writer = None                          # TensorBoard SummaryWriter

    def start(self) -> None: ...
    def step(self, epoch: int, step: int) -> dict: ...
    def get_peak_memory(self) -> dict: ...
    def log_to_tensorboard(self, epoch: int) -> None: ...
    def check_warning(self) -> list[str]: ...
```

### 2.2 监控指标

| 指标 | 来源 | 单位 | 说明 |
|------|------|------|------|
| `gpu_allocated` | `torch.cuda.memory_allocated()` | MB | 当前GPU显存分配 |
| `gpu_reserved` | `torch.cuda.memory_reserved()` | MB | 当前GPU显存保留 |
| `gpu_peak` | `torch.cuda.max_memory_allocated()` | MB | 本次峰值分配 |
| `gpu_util` | `nvidia-smi` | % | GPU利用率 |
| `ram_used` | `psutil.virtual_memory()` | MB | 系统内存使用 |
| `ram_percent` | `psutil.virtual_memory()` | % | 系统内存使用率 |

### 2.3 阈值告警机制

```python
def check_warning(self) -> list[str]:
    """检查是否超过阈值，返回告警信息列表"""
    warnings = []
    total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e6
    if self.peak_gpu_mb / total_gpu > self.warning_threshold:
        warnings.append(
            f"GPU memory {self.peak_gpu_mb:.0f}MB / {total_gpu:.0f}MB "
            f"({self.peak_gpu_mb/total_gpu*100:.1f}%) exceeds threshold"
        )
    ram = psutil.virtual_memory()
    if ram.percent > 90:
        warnings.append(f"System RAM usage: {ram.percent}%")
    return warnings
```

### 2.4 TensorBoard日志记录

```python
def log_to_tensorboard(self, epoch: int) -> None:
    if self.writer is None:
        return
    mem = self.get_peak_memory()
    self.writer.add_scalar("memory/gpu_peak_mb", mem["gpu_peak_mb"], epoch)
    self.writer.add_scalar("memory/gpu_allocated_mb", mem["gpu_allocated_mb"], epoch)
    self.writer.add_scalar("memory/ram_used_mb", mem["ram_used_mb"], epoch)
    self.writer.add_scalar("memory/ram_percent", mem["ram_percent"], epoch)
```

### 2.5 使用模式

```python
# 训练循环中集成
monitor = MemoryMonitor(warning_threshold=0.85)
monitor.start()

for epoch in range(num_epochs):
    for batch in dataloader:
        # ... 训练步骤 ...
        monitor.step(epoch, step)

    # 每epoch结束记录
    monitor.log_to_tensorboard(epoch)
    for w in monitor.check_warning():
        logger.warning(w)
```

---

## 3. 推理API规范

### 3.1 Predictor 接口

```python
from typing import Union, List, Dict, Optional
import numpy as np

class Match3Predictor:
    """三消模型推理接口"""

    def __init__(self, model, config, device="cuda", enforce_rules=True):
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.enforce_rules = enforce_rules  # 是否启用规则约束

    def predict(self, board: np.ndarray) -> dict:
        """
        单张棋盘推理

        Args:
            board: (H, W) int64 numpy数组，水果类型索引

        Returns:
            {
                "mask": np.ndarray,          # (H, W) float32 消除概率
                "pattern_types": np.ndarray, # (num_classes,) 可选，Pattern分类
                "metadata": {
                    "inference_time_ms": float,
                    "board_size": tuple,
                    "num_candidates": int,   # 候选消除格子数
                }
            }
        """

    def batch_predict(self, boards: List[np.ndarray]) -> List[dict]: ...

    def evaluate_accuracy(self, dataset, metric="iou") -> dict:
        """
        评估数据集准确率

        Returns:
            {
                "mean_iou": float,
                "pattern_accuracy": float,  # 可选
                "per_class_iou": dict,
                "total_samples": int,
            }
        """
```

### 3.2 输入输出详细规范

#### 输入

| 参数 | 类型 | 形状 | 值域 | 说明 |
|------|------|------|------|------|
| `board` | np.ndarray | `(H, W)` | `[0, num_fruit_types)` | 水果类型索引 |

#### 输出

| 字段 | 类型 | 形状 | 值域 | 说明 |
|------|------|------|------|------|
| `mask` | np.ndarray | `(H, W)` | `[0, 1]` | 消除概率 |
| `pattern_types` | np.ndarray | `(num_classes,)` | logit | Pattern分类logits（仅Transformer模型） |
| `metadata.inference_time_ms` | float | - | - | 推理耗时 |
| `metadata.num_candidates` | int | - | - | mask>threshold的格子数 |

### 3.3 后处理规则约束集成点

```python
def predict(self, board: np.ndarray) -> dict:
    # 1. numpy → tensor预处理
    tensor = self._preprocess(board).to(self.device)

    # 2. 模型推理
    with torch.no_grad():
        with autocast(enabled=self.config.get("mixed_precision", False)):
            output = self.model(tensor)

    # 3. 后处理: 规则约束层 (集成点)
    mask = output["mask"].sigmoid().cpu().numpy()
    if self.enforce_rules:
        # 调用 match3-pattern-core 的规则约束
        mask = enforce_match3_rules(mask, board)

    # 4. 组装结果
    result = {"mask": mask[0], "metadata": {...}}
    if "pattern_types" in output:
        result["pattern_types"] = output["pattern_types"].cpu().numpy()
    return result
```

> **集成点说明**: `enforce_rules=True` 时，推理结果会通过 `match3-pattern-core` Skill 定义的 `enforce_match3_rules` 函数进行后处理过滤。

---

## 4. CLI入口规范

### 4.1 命令行参数设计

```bash
python -m match3_toolkit \
    --mode train|eval|infer \           # 运行模式（必需）
    --config configs/unet_20x20.yaml \  # 配置文件路径（必需）
    --checkpoint checkpoints/best.pt \  # 检查点路径（eval/infer必需）
    --board data/test_board.npy \       # 输入棋盘（infer模式）
    --output_dir outputs/ \             # 输出目录
    --device cuda \                     # 计算设备
    --seed 42                           # 随机种子
```

### 4.2 配置JSON加载机制

```python
def load_config(config_path: str) -> Union[Match3UNetConfig, Match3HybridTransformerConfig]:
    """
    从YAML/JSON加载配置，根据arch字段自动选择配置类

    Args:
        config_path: 配置文件路径 (.yaml 或 .json)

    Returns:
        对应的配置实例
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    arch = raw.get("arch", "unet")
    if arch == "unet":
        return Match3UNetConfig(**raw)
    elif arch == "hybrid_transformer":
        return Match3HybridTransformerConfig(**raw)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
```

### 4.3 各模式执行流程

#### Train 模式

```
1. 加载配置文件 → 构建配置对象
2. 根据配置构建模型、优化器、损失函数
3. 加载数据集（train/val）
4. 初始化 MemoryMonitor、TensorBoard
5. for epoch in epochs:
   - train_one_epoch()
   - validate()
   - save_checkpoint() (如为best)
6. 保存最终检查点
```

#### Eval 模式

```
1. 加载配置 + 检查点
2. 构建模型并加载权重
3. 加载测试数据集
4. Predictor.evaluate_accuracy()
5. 输出指标报告
```

#### Infer 模式

```
1. 加载配置 + 检查点
2. 构建模型并加载权重
3. 加载输入board (.npy)
4. Predictor.predict() 或 batch_predict()
5. 保存结果mask + 可视化
```

---

## 5. 检查点管理规范

### 5.1 保存格式

```python
checkpoint = {
    # 必需字段
    "model_state_dict": model.state_dict(),        # 模型权重
    "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
    "config": config.__dict__,                      # 配置字典

    # 可选字段
    "epoch": epoch,                                 # 当前epoch
    "best_metric": best_iou,                        # 最佳指标值
    "scheduler_state_dict": scheduler.state_dict(),  # 学习率调度器
    "scaler_state_dict": scaler.state_dict(),       # AMP scaler
    "rng_state": torch.get_rng_state(),             # 随机状态
}
torch.save(checkpoint, "checkpoint.pt")
```

### 5.2 自动保存策略

| 策略 | 触发条件 | 文件名 | 保留数量 |
|------|---------|--------|---------|
| Best | 验证指标优于历史最佳 | `best_model.pt` | 1 |
| Final | 训练结束 | `final_model.pt` | 1 |
| 周期性 | 每N个epoch | `checkpoint_epoch_{N}.pt` | 最多3个 |
| 恢复 | 异常中断时 | `checkpoint_last.pt` | 1 |

```python
class CheckpointManager:
    """检查点管理器"""

    def __init__(self, output_dir, keep_last_n=3, monitor="val_iou", mode="max"):
        self.output_dir = output_dir
        self.keep_last_n = keep_last_n
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.periodic_checkpoints = []

    def save(self, checkpoint: dict, epoch: int, is_best: bool) -> None: ...
    def load(self, checkpoint_path: str) -> dict: ...
    def cleanup_old(self) -> None: ...
```

### 5.3 加载规范

```python
def load_checkpoint(model, checkpoint_path, device="cuda", strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # 恢复训练状态（如继续训练）
    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0)
    return model, epoch, best_metric
```

---

## 6. 项目目录结构规范

### 6.1 推荐的标准目录布局

```
match3-project/
├── configs/                        # 配置文件
│   ├── unet_20x20.yaml
│   ├── unet_50x50.yaml
│   └── hybrid_transformer_20x20.yaml
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 处理后数据
│   └── test_boards/                # 测试棋盘
├── checkpoints/                    # 模型检查点
│   ├── best_model.pt
│   ├── final_model.pt
│   └── checkpoint_last.pt
├── outputs/                        # 推理输出
│   ├── masks/                      # 预测mask
│   ├── visualizations/             # 可视化结果
│   └── metrics.json                # 评估指标
├── logs/                           # 日志
│   ├── tensorboard/                # TB事件文件
│   └── train.log                   # 训练日志
├── src/                            # 源代码
│   ├── models/                     # 模型定义
│   ├── data/                       # 数据加载
│   ├── training/                   # 训练逻辑
│   └── utils/                      # 工具函数
├── requirements.txt
└── README.md
```

### 6.2 目录说明

| 目录 | 用途 | 是否版本控制 |
|------|------|-------------|
| `configs/` | 模型配置文件 | 是 |
| `data/raw/` | 原始棋盘数据 | 否（大文件） |
| `data/processed/` | 预处理缓存 | 否 |
| `checkpoints/` | 模型权重 | 否 |
| `outputs/` | 推理结果 | 否 |
| `logs/` | 训练日志 | 否 |
| `src/` | 源代码 | 是 |

---

## 7. TensorBoard查看指南

### 7.1 启动命令

```bash
# 基本启动
tensorboard --logdir logs/tensorboard/

# 指定端口
tensorboard --logdir logs/tensorboard/ --port 6006

# 远程访问
 tensorboard --logdir logs/tensorboard/ --bind_all --port 6006
```

### 7.2 指标命名约定

```python
# 训练指标
writer.add_scalar("train/loss", train_loss, epoch)
writer.add_scalar("train/loss_mask", mask_loss, epoch)
writer.add_scalar("train/loss_pattern", pattern_loss, epoch)
writer.add_scalar("train/learning_rate", lr, epoch)

# 验证指标
writer.add_scalar("val/iou", val_iou, epoch)
writer.add_scalar("val/iou_per_class/class_0", iou_0, epoch)
writer.add_scalar("val/pattern_accuracy", pattern_acc, epoch)

# 系统指标
writer.add_scalar("system/gpu_memory_peak_mb", gpu_peak, epoch)
writer.add_scalar("system/gpu_utilization", gpu_util, epoch)

# 图像可视化
writer.add_image("val/pred_mask", pred_mask_vis, epoch)
writer.add_image("val/gt_mask", gt_mask_vis, epoch)
writer.add_image("val/board_overlay", overlay, epoch)
```

### 7.3 命名空间规范

| 前缀 | 用途 | 示例 |
|------|------|------|
| `train/` | 训练过程指标 | `train/loss`, `train/lr` |
| `val/` | 验证指标 | `val/iou`, `val/accuracy` |
| `test/` | 测试指标 | `test/iou`, `test/inference_time` |
| `system/` | 系统资源 | `system/gpu_memory`, `system/ram` |
| `histogram/` | 参数分布 | `histogram/conv1.weight` |

### 7.4 推荐查看的Scalar指标

| 指标名 | 关注目的 | 健康趋势 |
|--------|---------|---------|
| `train/loss` | 训练收敛 | 稳定下降 |
| `val/iou` | 验证性能 | 上升后平稳 |
| `val/iou_per_class/*` | 各类别性能 | 均衡无塌陷 |
| `system/gpu_memory_peak_mb` | 显存健康 | 稳定无持续增长 |
| `train/learning_rate` | 学习率调度 | 按计划衰减 |

### 7.5 推荐查看的Image

| 图像名 | 内容 | 用途 |
|--------|------|------|
| `val/pred_mask` | 预测mask热力图 | 检查预测质量 |
| `val/gt_mask` | 真值mask | 对比预测 |
| `val/board_overlay` | 棋盘+mask叠加 | 直观验证 |

---

## 8. 依赖与安装

```bash
pip install torch>=2.0.0 psutil>=5.9.0 tensorboard>=2.13.0
```

### 8.1 基础Skill依赖

| Skill | 用途 |
|-------|------|
| `match3-training-foundation` | 训练循环、优化器、损失函数、指标计算基础 |

---

## 9. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2024-XX-XX | 初始版本：显存监控 + 推理API + CLI + 检查点管理 |

---

<!-- file-tags: skill, match3, utils, monitoring, inference, cli, checkpoint, tensorboard -->
