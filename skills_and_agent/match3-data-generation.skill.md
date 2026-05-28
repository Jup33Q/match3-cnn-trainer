<!--
  SKILL: match3-data-generation
  DESC: 三消棋盘数据生成策略规范，定义程序化数据生成、难度控制和Pattern注入标准
  AUTHOR: AI Skill System
  VERSION: 1.0.0
  DEPENDS: numpy>=1.24.0, torch>=2.0.0
  COMPAT: Python 3.9+
-->

# match3-data-generation

> 三消棋盘数据生成策略规范
>
> 版本: 1.0.0 | 依赖: numpy>=1.24.0, torch>=2.0.0

---

## 1. 核心概念

### 1.1 三种难度类型

生成数据集时，每个样本属于以下三种难度类型之一，难度差异决定了Pattern的密度和棋盘结构：

| 难度类型 | 说明 | Pattern密度 | mask特征 |
|----------|------|-------------|----------|
| **easy** | 完全随机棋盘 | 极低（自然形成少量H3/V3） | mask几乎全为0（NONE） |
| **hard** | 大量"接近三消"结构 | 零（无完整Pattern） | mask全为0，但存在大量两连结构 |
| **positive** | 注入可控三消Pattern | 可控（按密度参数注入） | mask含明确的Pattern标注 |

> **设计意图**：easy样本训练模型区分随机噪声；hard样本训练模型识别"准Pattern"避免误判；positive样本提供正例学习信号。

---

> **注意**: `match3_cnn_trainer` 项目已统一改用 **One-Hot 颜色通道** 编码（`max_fruit_types=16` 通道），FruitRoPE 保留在代码库中但不再被训练流程使用。

## 2. FruitRoPE 编码

### 2.1 设计动机

传统 one-hot 编码将 fruit 类型表示为 `(num_fruit_types, H, W)`，但存在以下问题：
- **固定类型数**：模型输入通道数绑死 `num_fruit_types`，无法动态扩展
- **高维稀疏**：12 种 fruit 需要 12 个通道，信息密度低
- **无相似性**：不同类型之间无内在几何关系

**FruitRoPE** 使用 1D 旋转位置编码（Rotary Position Embedding）将 fruit ID 映射到固定低维空间：

```
fruit_id (0~max_types-1) → RoPE embedding (dim=32)
```

### 2.2 编码原理

```python
class FruitRoPE:
    def __init__(self, dim: int = 32, max_types: int = 16, base: float = 10000.0):
        positions = torch.arange(max_types).float()          # (max_types,)
        i = torch.arange(0, dim, 2).float()                  # (dim//2,)
        theta = base ** (-2 * i / dim)                       # (dim//2,)
        angles = positions.unsqueeze(1) * theta.unsqueeze(0)  # (max_types, dim//2)
        emb = torch.zeros(max_types, dim)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        self.emb = emb  # (max_types, dim)

    def encode(self, board: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(board).long().clamp(0, self.max_types - 1)
        return self.emb[t].permute(2, 0, 1)  # (dim, H, W)
```

### 2.3 特性与优势

| 特性 | one-hot | FruitRoPE |
|------|---------|-----------|
| 输入维度 | `num_fruit_types` (变长) | `fruit_embed_dim=32` (固定) |
| 动态类型支持 | ❌ 需改模型 | ✅ 5~16 种自动适配 |
| 类型间相似性 | ❌ 正交无关 | ✅ 角度相近则相似 |
| 模型兼容性 | 旧检查点通道不匹配 | 零填充可迁移 |

> **配置参数**：`fruit_embed_dim=32`, `max_fruit_types=16`

---

## 3. 难度比例配置

### 2.1 默认配置

```
difficulty_ratio = 0.3 : 0.4 : 0.3
                  (easy : hard : positive)
```

### 2.2 配置说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `easy_ratio` | float | 0.3 | easy样本在总数据集中的比例 |
| `hard_ratio` | float | 0.4 | hard样本在总数据集中的比例 |
| `positive_ratio` | float | 0.3 | positive样本在总数据集中的比例 |

> **调优建议**：若模型Recall偏低，可适当提高positive_ratio；若模型Precision偏低，可适当提高hard_ratio增强负例学习。

---

## 3. Pattern注入策略

### 3.1 注入密度规范

```
单个棋盘注入Pattern数量范围:
    min_patterns = (h × w) // 500
    max_patterns = (h × w) // 300
```

| 棋盘尺寸 | 最小Pattern数 | 最大Pattern数 |
|----------|--------------|--------------|
| 8 × 8 = 64 | 1 | 1 |
| 12 × 12 = 144 | 1 | 2 |
| 20 × 20 = 400 | 1 | 2 |
| 30 × 30 = 900 | 2 | 3 |
| 64 × 64 = 4096 | 9 | 14 |

> **注意**：注入时需避免Pattern之间的重叠冲突。若新注入的Pattern与已有Pattern重叠，应丢弃并重新尝试注入。

### 3.2 直线型Pattern注入（H3-H5, V3-V5）

**注入算法**：
1. 随机选择起始位置 `(r, c)` 和方向（横向/纵向）
2. 随机选择长度（3/4/5），确保不越界
3. 随机选择元素类型 `t`
4. 将对应位置的元素设置为类型 `t`
5. 使用 `PatternDetector` 验证注入结果

**边界条件**：
- 注入区域不得与已有Pattern区域重叠
- 注入后不得产生非预期的额外Pattern（如注入H3时意外形成L型）

### 3.3 L型Pattern注入

**注入结构**：
- 选择拐角位置 `(r, c)` 作为L型拐角点
- 从拐角点向一个方向（水平）延伸2个相同类型元素
- 从拐角点向另一个方向（垂直）延伸2个相同类型元素
- 确保5个位置均在棋盘内且不与已有Pattern冲突

**旋转方向**：随机选择4种方向之一（左上、右上、左下、右下）

### 3.4 T型Pattern注入

**注入结构**：
- 选择中心位置 `(r, c)` 作为T交叉点
- 沿主轴（如横向）放置3个连续相同元素，中心在 `(r, c)`
- 沿次轴（如垂直向下）从中心延伸2个相同元素
- 总元素数 = 5

**方向选择**：随机选择4种T方向之一

### 3.5 十字型Pattern（CROSS）注入

**注入结构**：
- 选择中心位置 `(r, c)`
- 横向放置3个连续相同元素，`(r, c)` 为中心
- 纵向放置3个连续相同元素，`(r, c)` 为中心
- 总元素数 = 5（中心点共享）

> **优先级处理**：CROSS型Pattern具有最高优先级，注入后该区域的格点不再参与其他Pattern分配。

---

## 4. Stage5 动态生成策略

### 4.1 Stage5 概述

Stage5 是课程学习的最终阶段，旨在让模型完全泛化到动态变化的棋盘环境：

| 参数 | 动态范围 | 说明 |
|------|----------|------|
| 棋盘尺寸 | 10 ~ 50 (随机) | 每次生成随机尺寸 |
| Fruit种类数 | 5 ~ 12 (随机) | 超出固定6种的限制 |
| Match长度 | 5 ~ 8 (随机) | 更长链式消除 |
| 输入编码 | RoPE (32维) | 与动态fruit种类数解耦 |

### 4.2 动态参数覆盖

`Match3BoardGenerator.generate_board()` 支持通过可选参数临时覆盖配置：

```python
board, mask, pattern_map = generator.generate_board(
    size=30,
    difficulty="positive",
    num_fruit_types=8,      # 临时覆盖: 使用8种fruit
    min_match_length=5,     # 临时覆盖: 最小match长度5
    max_match_length=7      # 临时覆盖: 最大注入长度7
)
```

### 4.3 随机尺寸嵌入策略

当 `stage_size=-1` 或 `-2` 时，Dataset 生成随机尺寸棋盘后，将其嵌入到 `max_size x max_size` (如50x50) 的固定画布中随机位置，保证模型输入尺寸恒定：

```python
actual_size = random.randint(10, max_size)
# 生成 actual_size x actual_size 的棋盘
# 嵌入到 max_size x max_size 画布的随机位置
```

> **目的**：模型输入始终为 `(B, fruit_embed_dim, max_size, max_size)`，便于批处理和卷积运算。

---

## 5. Near-Miss棋盘生成策略

### 4.1 概念说明

Near-Miss（接近三消）棋盘是hard样本的核心。棋盘中包含大量"差一步即可三消"的两连结构，但**不存在任何完整的三消Pattern**。

### 4.2 生成算法

```
输入: board_size (h, w), num_types
输出: near_miss_board (h, w)

步骤:
1. 生成完全随机棋盘
2. 随机注入多个"两连"结构（横向或纵向的两个连续相同元素）
3. 在两连结构的延伸方向（第3个位置）放置不同类型的元素，阻断三消
4. 重复步骤2-3直到达到目标near-miss密度
5. 使用PatternDetector验证棋盘上无任何完整Pattern
6. 若存在意外Pattern，回滚并重试
```

### 4.3 Near-Miss密度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `near_miss_density` | float | 0.15 | 棋盘上两连结构的目标占比（格点级别） |
| `block_type_strategy` | str | "random_different" | 阻断元素选择策略（随机不同类型） |

---

## 5. 程序化标注流程

### 5.1 完整生成流程

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. 生成棋盘     │ → │  2. 注入Pattern   │ → │  3. 精确标注      │
│  (随机/near-miss)│    │  (仅positive类型) │    │  (mask + type)   │
└─────────────────┘    └──────────────────┘    └──────────────────┘
         ↓                      ↓                      ↓
    ┌──────────┐          ┌──────────┐          ┌──────────┐
    │  board   │          │  board   │          │  board   │
    │  (H, W)  │          │  (H, W)  │          │  (H, W)  │
    └──────────┘          └──────────┘          └──────────┘
                                                 ┌──────────┐
                                                 │  mask    │
                                                 │ (H, W)   │
                                                 │  int数组  │
                                                 └──────────┘
                                                 ┌──────────┐
                                                 │ pattern  │
                                                 │  types   │
                                                 │ (H, W)   │
                                                 └──────────┘
```

### 5.2 标注规范

- **mask**：`(H, W)` 的二值数组，`1` 表示该格点属于某个Pattern，`0` 表示NONE
- **pattern_type**：`(H, W)` 的整型数组，值域对应 `PatternType` 枚举值
- **一致性要求**：若 `pattern_type[i,j] != NONE`，则 `mask[i,j] == 1` 必须成立

---

## 6. 配置接口规范

### 6.1 数据生成配置（Python dataclass伪代码）

```python
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class Match3DataConfig:
    """三消棋盘数据生成配置"""

    # === 棋盘参数 ===
    board_size: Tuple[int, int] = (20, 20)   # (height, width)
    num_element_types: int = 5                # 元素类型数（如5种宝石颜色）

    # === 难度比例 ===
    easy_ratio: float = 0.3                   # 完全随机棋盘比例
    hard_ratio: float = 0.4                   # near-miss棋盘比例
    positive_ratio: float = 0.3               # 注入Pattern棋盘比例

    # === Pattern注入密度 ===
    pattern_density_min: float = 1/500        # 最小Pattern密度
    pattern_density_max: float = 1/300        # 最大Pattern密度

    # === 各类Pattern注入权重 ===
    pattern_weights: dict = field(default_factory=lambda: {
        "straight": 0.5,    # 直线型（H3-H5, V3-V5）
        "L": 0.2,           # L型
        "T": 0.2,           # T型
        "CROSS": 0.1,       # 十字型
    })

    # === Near-Miss参数 ===
    near_miss_density: float = 0.15           # 两连结构目标密度

    # === 随机种子 ===
    seed: Optional[int] = None

@dataclass
class Match3DatasetConfig:
    """数据集整体配置"""
    data_config: Match3DataConfig
    num_samples: int = 10000                  # 总样本数
    batch_size: int = 32                      # DataLoader batch size
    num_workers: int = 4                      # 数据加载并行度
```

---

## 7. 与match3-pattern-core的协作关系

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   match3-data-generation │  ───►  │   match3-pattern-core    │
│   （数据生成器）          │  依赖   │   （Pattern检测引擎）    │
├─────────────────────────┤         ├─────────────────────────┤
│ • 生成棋盘布局           │         │ • PatternType枚举定义    │
│ • 注入可控Pattern        │         │ • Pattern优先级规则      │
│ • 控制难度分布           │         │ • PatternDetector        │
│ • 生成黄金标准mask       │  ───►  │   (detect & 标注)       │
└─────────────────────────┘  调用    └─────────────────────────┘
```

### 7.1 依赖说明

- `match3-data-generation` **依赖** `match3-pattern-core` 提供的 `PatternType` 枚举和 `PatternDetector` 检测器
- 数据生成器在注入Pattern后，必须调用 `PatternDetector.detect()` 进行黄金标准标注
- 不得自行实现Pattern判定逻辑，避免与核心引擎的规则不一致

### 7.2 数据流

1. 数据生成器创建棋盘（随机 / near-miss / Pattern注入）
2. 调用 `PatternDetector` 的 `detect()` 方法获取 `pattern_mask`
3. 将 `board`、`mask`、`pattern_type` 打包为训练样本
4. 输出给 `DataLoader` 供模型训练使用

---

## 8. 版本兼容性

| 版本 | 兼容范围 | 说明 |
|------|----------|------|
| 1.0.0 | Python 3.9+, NumPy 1.24+, PyTorch 2.0+ | 初始版本，支持3种难度类型 |

---

<!-- SKILL-REF: match3-data-generation -->
