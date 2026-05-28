# Agent: match3-cnn-trainer-agent
# Description: U-Net三消Pattern识别模型完整训练工作流，包含数据生成、模型构建、训练循环和推理
# Version: 1.0.0
# Dependencies: match3-pattern-core, match3-data-generation, match3-cnn-unet-trainer, match3-training-foundation, match3-shared-utils
# Stack: PyTorch >=2.0.0, Python >=3.9

---

## 概述

本Agent提供完整的U-Net三消Pattern识别模型训练工作流，包含：
- 参数化配置管理
- Pattern类型检测（H/V/L/T/Cross）
- 程序化棋盘数据生成
- U-Net模型架构（大卷积核 + 膨胀卷积 + 可选Mamba2D层）
- Dice + Focal + Boundary组合损失
- 规则后处理（连通性约束）
- 课程学习训练引擎
- 推理与评估接口
- CLI主入口

---

## 执行步骤

### 1. 安装依赖

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 pillow>=9.5.0 tqdm>=4.65.0 tensorboard>=2.13.0 albumentations>=1.3.0 opencv-python>=4.7.0
```

### 2. 运行训练

```bash
# 默认配置训练 (100x100棋盘, 50k样本)
python main.py --mode train

# 自定义配置训练
python main.py --mode train --config custom_config.json
```

### 3. 运行评估

```bash
python main.py --mode eval --checkpoint ./checkpoints/match3_unet_epochfinal.pt
```

### 4. 运行推理

```bash
python main.py --mode infer --checkpoint ./checkpoints/match3_unet_epochfinal.pt --board board.npy
```

---

## 代码模块

### 模块1: config.py — 参数化配置接口

```python
"""config.py - 所有可调参数集中管理"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Match3Config:
    """三消 Pattern 识别模型配置"""

    # --- 数据参数 ---
    board_size: int = 50                     # 棋盘尺寸 (50x50)
    num_fruit_types: int = 6                 # 水果种类数
    train_samples: int = 50000               # 训练样本数
    val_samples: int = 5000                  # 验证样本数

    # --- 数据生成策略 ---
    difficulty_ratio: Tuple[float, float, float] = (0.3, 0.4, 0.3)  # 简单:困难:正例
    min_match_length: int = 3                # 最小消除长度
    max_match_length: int = 5                # 最大消除长度 (注入时)

    # --- 模型参数 ---
    encoder_backbone: str = "custom_cnn"     # 可选: custom_cnn | mobilenetv3 | segformer
    initial_kernel_size: int = 7             # 首层大卷积核 (捕获局部连续性)
    base_channels: int = 64                  # 基础通道数
    num_encoder_blocks: int = 4              # Encoder 块数量
    use_dilation: bool = True                # 是否使用膨胀卷积
    dilation_rates: List[int] = field(default_factory=lambda: [1, 2, 4, 8])  # 膨胀率序列

    # --- 训练参数 ---
    batch_size: int = 80
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"             # cosine | plateau | step
    warmup_epochs: int = 5

    # --- 损失函数权重 ---
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    boundary_weight: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # --- 课程学习 ---
    curriculum_enabled: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [10, 25, 50, -1, -3, -2])  # 棋盘尺寸递进 (-1=正方形随机10~50过渡, -3=长方形随机10~50, -2=stage5长方形随机+颜色随机)
    curriculum_epochs_per_stage: int = 20

    # --- 后处理 ---
    mask_threshold: float = 0.5
    enforce_connectivity: bool = True        # 强制连通性约束

    # --- 系统参数 ---
    device: str = "auto"                     # auto | cuda | cpu | mps
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 10                     # 每N epoch保存

    # --- Fruit One-Hot 编码 ---
    fruit_embed_dim: int = 32                # (已废弃) 原正n边形顶点编码维度，现统一使用One-Hot
    max_fruit_types: int = 16                # 正n边形顶点3通道数（stage5动态5~12种通过前n个通道表达）
    stage5_match_length_range: Tuple[int, int] = (5, 8)   # stage5 match长度范围
    stage5_fruit_range: Tuple[int, int] = (5, 12)         # stage5 fruit种类范围
```

### 模块2: data/pattern_types.py — Pattern类型定义与检测器

```python
"""data/pattern_types.py - Pattern类型枚举 + PatternDetector完整实现"""
from enum import Enum
from typing import Tuple, List, Dict, Set
import numpy as np


class PatternType(Enum):
    """三消Pattern类型"""
    NONE = 0
    H3 = 1      # 横向3连
    H4 = 2      # 横向4连
    H5 = 3      # 横向5连
    V3 = 4      # 纵向3连
    V4 = 5      # 纵向4连
    V5 = 6      # 纵向5连
    L = 7       # L型 (2x3或3x2 corner)
    T = 8       # T型 (3x2或2x3 with branch)
    CROSS = 9   # 十字型 (3x3 center)


class PatternDetector:
    """
    Pattern检测器: 识别棋盘上的所有消除Pattern及其类型
    输出每个格点所属的"最大"Pattern类型
    """

    PATTERN_NAMES = {
        0: "NONE", 1: "H3", 2: "H4", 3: "H5",
        4: "V3", 5: "V4", 6: "V5",
        7: "L", 8: "T", 9: "CROSS"
    }

    # Pattern优先级 (用于确定"最大"Pattern)
    PATTERN_PRIORITY = {
        PatternType.CROSS: 100,
        PatternType.T: 80,
        PatternType.L: 60,
        PatternType.H5: 50,
        PatternType.V5: 50,
        PatternType.H4: 40,
        PatternType.V4: 40,
        PatternType.H3: 30,
        PatternType.V3: 30,
        PatternType.NONE: 0
    }

    @staticmethod
    def detect_all_patterns(board: np.ndarray, min_length: int = 3) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测棋盘上所有Pattern

        Returns:
            type_map: (H, W) int array, 每个格点的Pattern类型
            patterns: list of dict, 每个检测到的Pattern信息
        """
        h, w = board.shape
        type_map = np.full((h, w), PatternType.NONE.value, dtype=np.int32)
        patterns = []

        # Step 1: 检测所有直线型Pattern (H3-H5, V3-V5)
        line_patterns = PatternDetector._detect_lines(board, min_length)
        patterns.extend(line_patterns)

        # Step 2: 检测L型Pattern
        l_patterns = PatternDetector._detect_l_shapes(board, line_patterns)
        patterns.extend(l_patterns)

        # Step 3: 检测T型Pattern
        t_patterns = PatternDetector._detect_t_shapes(board, line_patterns)
        patterns.extend(t_patterns)

        # Step 4: 检测十字型Pattern
        cross_patterns = PatternDetector._detect_crosses(board, line_patterns)
        patterns.extend(cross_patterns)

        # Step 5: 为每个格点分配"最大"Pattern类型
        type_map = PatternDetector._assign_max_patterns(h, w, patterns)

        return type_map, patterns

    @staticmethod
    def _detect_lines(board: np.ndarray, min_length: int) -> List[Dict]:
        """检测横向和纵向直线Pattern"""
        h, w = board.shape
        patterns = []

        # 横向检测
        for y in range(h):
            x = 0
            while x < w:
                color = board[y, x]
                length = 1
                while x + length < w and board[y, x + length] == color:
                    length += 1

                if length >= min_length:
                    pattern_type = getattr(PatternType, f"H{min(length, 5)}", PatternType.H5)
                    coords = [(y, xx) for xx in range(x, x + length)]
                    patterns.append({
                        "type": pattern_type,
                        "type_id": pattern_type.value,
                        "color": int(color),
                        "coords": coords,
                        "priority": PatternDetector.PATTERN_PRIORITY[pattern_type]
                    })
                x += length

        # 纵向检测
        for x in range(w):
            y = 0
            while y < h:
                color = board[y, x]
                length = 1
                while y + length < h and board[y + length, x] == color:
                    length += 1

                if length >= min_length:
                    pattern_type = getattr(PatternType, f"V{min(length, 5)}", PatternType.V5)
                    coords = [(yy, x) for yy in range(y, y + length)]
                    patterns.append({
                        "type": pattern_type,
                        "type_id": pattern_type.value,
                        "color": int(color),
                        "coords": coords,
                        "priority": PatternDetector.PATTERN_PRIORITY[pattern_type]
                    })
                y += length

        return patterns

    @staticmethod
    def _detect_l_shapes(board: np.ndarray, line_patterns: List[Dict]) -> List[Dict]:
        """
        检测L型Pattern:
        - 一个横向3连 + 一个纵向3连，共享一个角点
        - 形状:  口 或  口口
                口      口
                口
        - 总共5格 (3+3-1)
        """
        h, w = board.shape
        patterns = []

        # 获取所有H3和V3
        h3_patterns = [p for p in line_patterns if p["type"] == PatternType.H3]
        v3_patterns = [p for p in line_patterns if p["type"] == PatternType.V3]

        for h3 in h3_patterns:
            h3_coords = set(h3["coords"])
            h3_color = h3["color"]

            for v3 in v3_patterns:
                if v3["color"] != h3_color:
                    continue

                v3_coords = set(v3["coords"])
                intersection = h3_coords & v3_coords

                # L型: 共享恰好1个格点 (角点)
                if len(intersection) == 1:
                    all_coords = list(h3_coords | v3_coords)
                    # 验证是L型: 总共5格 (3+3-1=5)
                    if len(all_coords) == 5:
                        patterns.append({
                            "type": PatternType.L,
                            "type_id": PatternType.L.value,
                            "color": h3_color,
                            "coords": all_coords,
                            "priority": PatternDetector.PATTERN_PRIORITY[PatternType.L],
                            "corner": list(intersection)[0]
                        })

        return patterns

    @staticmethod
    def _detect_t_shapes(board: np.ndarray, line_patterns: List[Dict]) -> List[Dict]:
        """
        检测T型Pattern:
        - 横向3连 + 纵向3连，共享中心格点
        - 形状:  口口口  或   口
                 口          口口
                 口           口
        """
        patterns = []
        h3_list = [p for p in line_patterns if p["type"] == PatternType.H3]
        v3_list = [p for p in line_patterns if p["type"] == PatternType.V3]

        for h3 in h3_list:
            h3_coords = set(h3["coords"])
            h3_color = h3["color"]

            for v3 in v3_list:
                if v3["color"] != h3_color:
                    continue

                v3_coords = set(v3["coords"])
                intersection = h3_coords & v3_coords

                # T型: 共享恰好1个格点，且该点是某一条线的中点
                if len(intersection) == 1:
                    corner = list(intersection)[0]
                    all_coords = list(h3_coords | v3_coords)

                    # 检查是否形成T (不是L或十字)
                    if len(all_coords) == 5:
                        # T型: 一条线的中点与另一条线连接
                        h3_y_coords = [c[0] for c in h3["coords"]]
                        v3_x_coords = [c[1] for c in v3["coords"]]

                        is_h3_center = (corner[1] == sorted(v3_x_coords)[1])
                        is_v3_center = (corner[0] == sorted(h3_y_coords)[1])

                        if is_h3_center or is_v3_center:
                            patterns.append({
                                "type": PatternType.T,
                                "type_id": PatternType.T.value,
                                "color": h3_color,
                                "coords": all_coords,
                                "priority": PatternDetector.PATTERN_PRIORITY[PatternType.T],
                                "center": corner
                            })

        return patterns

    @staticmethod
    def _detect_crosses(board: np.ndarray, line_patterns: List[Dict]) -> List[Dict]:
        """
        检测十字型Pattern:
        - 横向3连 + 纵向3连，共享中心格点，且中心是两条线的中点
        - 形状:   口
                 口口口
                  口
        """
        patterns = []
        h3_list = [p for p in line_patterns if p["type"] == PatternType.H3]
        v3_list = [p for p in line_patterns if p["type"] == PatternType.V3]

        for h3 in h3_list:
            h3_coords = set(h3["coords"])
            h3_color = h3["color"]
            h3_y = h3["coords"][0][0]
            h3_xs = sorted([c[1] for c in h3["coords"]])

            for v3 in v3_list:
                if v3["color"] != h3_color:
                    continue

                v3_coords = set(v3["coords"])
                v3_x = v3["coords"][0][1]
                v3_ys = sorted([c[0] for c in v3["coords"]])

                intersection = h3_coords & v3_coords

                # 十字: 共享中心点，且中心是两条线的中点
                if len(intersection) == 1:
                    center = list(intersection)[0]
                    all_coords = list(h3_coords | v3_coords)

                    if len(all_coords) == 5:
                        # 检查中心是否是两条线的中点
                        is_h_center = center[1] == h3_xs[1]  # 横向中点
                        is_v_center = center[0] == v3_ys[1]  # 纵向中点

                        if is_h_center and is_v_center:
                            patterns.append({
                                "type": PatternType.CROSS,
                                "type_id": PatternType.CROSS.value,
                                "color": h3_color,
                                "coords": all_coords,
                                "priority": PatternDetector.PATTERN_PRIORITY[PatternType.CROSS],
                                "center": center
                            })

        return patterns

    @staticmethod
    def _assign_max_patterns(h: int, w: int, patterns: List[Dict]) -> np.ndarray:
        """
        为每个格点分配优先级最高的Pattern类型
        如果一个格点属于多个Pattern，选择优先级最高的
        """
        type_map = np.full((h, w), PatternType.NONE.value, dtype=np.int32)
        priority_map = np.zeros((h, w), dtype=np.int32)

        for pattern in patterns:
            p_type_id = pattern["type_id"]
            priority = pattern["priority"]

            for (y, x) in pattern["coords"]:
                if priority > priority_map[y, x]:
                    priority_map[y, x] = priority
                    type_map[y, x] = p_type_id

        return type_map
```

### 模块3: data/data_generator.py — 数据生成引擎

```python
"""data/data_generator.py - 程序化生成三消棋盘与消除Mask"""
import random
import torch
import numpy as np
from typing import Tuple

from config import Match3Config
from data.pattern_types import PatternDetector, PatternType


class FruitRoPE:
    """使用1D RoPE (Rotary Position Embedding) 编码fruit类型"""

    def __init__(self, dim: int = 32, max_types: int = 16, base: float = 10000.0):
        self.dim = dim
        self.max_types = max_types
        positions = torch.arange(max_types).float()          # (max_types,)
        i = torch.arange(0, dim, 2).float()                  # (dim // 2,)
        theta = base ** (-2 * i / dim)                       # (dim // 2,)
        angles = positions.unsqueeze(1) * theta.unsqueeze(0)  # (max_types, dim // 2)
        emb = torch.zeros(max_types, dim)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        self.emb = emb  # (max_types, dim)

    def encode(self, board: np.ndarray) -> torch.Tensor:
        """将fruit ID board编码为RoPE embedding
        Args:
            board: (H, W) int array, fruit type IDs
        Returns:
            (dim, H, W) float tensor
        """
        t = torch.from_numpy(board).long().clamp(0, self.max_types - 1)
        return self.emb[t].permute(2, 0, 1)  # (dim, H, W)


class Match3BoardGenerator:
    """程序化生成三消棋盘与消除 Mask"""

    def __init__(self, config: Match3Config):
        self.cfg = config
        self.rng = np.random.RandomState(config.seed)
        self.detector = PatternDetector()

    def generate_board(self, size: int, difficulty: str = "mixed",
                       num_fruit_types: int = None,
                       min_match_length: int = None,
                       max_match_length: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成棋盘和对应的三消 Mask

        Args:
            size: 棋盘尺寸 (size x size)
            difficulty: "easy" | "hard" | "positive"
            num_fruit_types: 临时覆盖的fruit种类数 (None=使用config)
            min_match_length: 临时覆盖的最小消除长度 (None=使用config)
            max_match_length: 临时覆盖的最大注入长度 (None=使用config)

        Returns:
            board: (size, size) int array, 每个格子的水果类型 [0, num_fruit_types)
            mask: (size, size) bool array, 1表示该格点参与三消
            pattern_types: (size, size) int array, Pattern类型ID
        """
        old_num = self.cfg.num_fruit_types
        old_min = self.cfg.min_match_length
        old_max = self.cfg.max_match_length

        try:
            if num_fruit_types is not None:
                self.cfg.num_fruit_types = num_fruit_types
            if min_match_length is not None:
                self.cfg.min_match_length = min_match_length
            if max_match_length is not None:
                self.cfg.max_match_length = max_match_length

            if difficulty == "easy":
                board = self.rng.randint(0, self.cfg.num_fruit_types, (size, size))
                pattern_map, patterns = self.detector.detect_all_patterns(board, self.cfg.min_match_length)
                mask = (pattern_map != PatternType.NONE.value)

            elif difficulty == "hard":
                board = self._generate_near_miss_board(size)
                mask = np.zeros((size, size), dtype=np.bool_)
                pattern_map = np.full((size, size), PatternType.NONE.value, dtype=np.int32)

            else:  # "positive"
                board = self.rng.randint(0, self.cfg.num_fruit_types, (size, size))
                board, mask, pattern_map = self._inject_match_patterns(board)

            return board, mask, pattern_map
        finally:
            self.cfg.num_fruit_types = old_num
            self.cfg.min_match_length = old_min
            self.cfg.max_match_length = old_max

    def _generate_near_miss_board(self, size: int) -> np.ndarray:
        """生成含大量接近三消但无实际消除的棋盘"""
        board = self.rng.randint(0, self.cfg.num_fruit_types, (size, size))

        # 注入两连结构
        num_patterns = size * 2
        for _ in range(num_patterns):
            y, x = self.rng.randint(0, size, 2)
            color = self.rng.randint(0, self.cfg.num_fruit_types)

            if self.rng.random() < 0.5 and x + 1 < size:
                board[y, x] = color
                board[y, x+1] = color
                if x + 2 < size:
                    board[y, x+2] = (color + 1) % self.cfg.num_fruit_types
                if x - 1 >= 0:
                    board[y, x-1] = (color + 2) % self.cfg.num_fruit_types
            elif y + 1 < size:
                board[y, x] = color
                board[y+1, x] = color
                if y + 2 < size:
                    board[y+2, x] = (color + 1) % self.cfg.num_fruit_types
                if y - 1 >= 0:
                    board[y-1, x] = (color + 2) % self.cfg.num_fruit_types

        return board

    def _inject_match_patterns(self, board: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在随机棋盘上注入三消 pattern (横向/纵向直线, L, T, Cross)"""
        h, w = board.shape
        num_injections = max(3, (h * w) // 400)

        for _ in range(num_injections):
            color = self.rng.randint(0, self.cfg.num_fruit_types)
            pattern_choice = self.rng.choice(["h", "v", "l", "t", "cross"], p=[0.3, 0.3, 0.15, 0.15, 0.1])

            if pattern_choice == "h":
                self._inject_horizontal(board, h, w, color)
            elif pattern_choice == "v":
                self._inject_vertical(board, h, w, color)
            elif pattern_choice == "l":
                self._inject_l_shape(board, h, w, color)
            elif pattern_choice == "t":
                self._inject_t_shape(board, h, w, color)
            elif pattern_choice == "cross":
                self._inject_cross(board, h, w, color)

        # 使用PatternDetector精确标注
        pattern_map, _ = self.detector.detect_all_patterns(board, self.cfg.min_match_length)
        mask = (pattern_map != PatternType.NONE.value)
        return board, mask, pattern_map

    def _inject_horizontal(self, board, h, w, color):
        """注入横向直线Pattern"""
        length = self.rng.randint(self.cfg.min_match_length, self.cfg.max_match_length + 1)
        y = self.rng.randint(0, h)
        x = self.rng.randint(0, w - length + 1)
        board[y, x:x+length] = color

    def _inject_vertical(self, board, h, w, color):
        """注入纵向直线Pattern"""
        length = self.rng.randint(self.cfg.min_match_length, self.cfg.max_match_length + 1)
        x = self.rng.randint(0, w)
        y = self.rng.randint(0, h - length + 1)
        board[y:y+length, x] = color

    def _inject_l_shape(self, board, h, w, color):
        """注入L型: 横向3 + 纵向3，共享角点"""
        if h < 3 or w < 3:
            return
        corner_y = self.rng.randint(1, h - 1)
        corner_x = self.rng.randint(1, w - 1)
        h_dir = self.rng.choice([-1, 1])
        v_dir = self.rng.choice([-1, 1])
        hx_start = min(corner_x, corner_x + h_dir * 2)
        hx_end = max(corner_x, corner_x + h_dir * 2)
        if 0 <= hx_start < hx_end < w:
            board[corner_y, hx_start:hx_end+1] = color
        vy_start = min(corner_y, corner_y + v_dir * 2)
        vy_end = max(corner_y, corner_y + v_dir * 2)
        if 0 <= vy_start < vy_end < h:
            board[vy_start:vy_end+1, corner_x] = color

    def _inject_t_shape(self, board, h, w, color):
        """注入T型: 横向3 + 纵向3，中心连接"""
        if h < 3 or w < 3:
            return
        center_y = self.rng.randint(1, h - 1)
        center_x = self.rng.randint(1, w - 1)
        hx_start = max(0, center_x - 1)
        hx_end = min(w - 1, center_x + 1)
        board[center_y, hx_start:hx_end+1] = color
        if self.rng.random() < 0.5:
            vy_end = min(h - 1, center_y + 2)
            board[center_y:vy_end+1, center_x] = color
        else:
            vy_start = max(0, center_y - 2)
            board[vy_start:center_y+1, center_x] = color

    def _inject_cross(self, board, h, w, color):
        """注入十字型: 横向3 + 纵向3，中心交叉"""
        if h < 3 or w < 3:
            return
        center_y = self.rng.randint(1, h - 1)
        center_x = self.rng.randint(1, w - 1)
        hx_start = max(0, center_x - 1)
        hx_end = min(w - 1, center_x + 1)
        board[center_y, hx_start:hx_end+1] = color
        vy_start = max(0, center_y - 1)
        vy_end = min(h - 1, center_y + 1)
        board[vy_start:vy_end+1, center_x] = color


class Match3Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset 封装，支持预加载"""

    def __init__(self, config: Match3Config, num_samples: int, stage_size: int = None):
        self.cfg = config
        self.num_samples = num_samples
        self.size = stage_size if stage_size is not None else config.board_size
        self.random_size = (stage_size == -1 or stage_size == -2)
        self.stage5_mode = (stage_size == -2)
        self.max_size = config.board_size
        self.generator = Match3BoardGenerator(config)
        self.fruit_rope = FruitRoPE(config.fruit_embed_dim, config.max_fruit_types)

        # 预生成所有数据 (内存允许时) 或 动态生成
        self.preload = num_samples <= 10000 and not self.random_size and not self.stage5_mode
        if self.preload:
            self.data = [self._generate_one() for _ in range(num_samples)]

    def _generate_one(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成单个样本，返回 (board_onehot, mask, fruit_ids)"""
        # 按难度比例采样
        difficulties = ["easy", "hard", "positive"]
        weights = self.cfg.difficulty_ratio
        difficulty = np.random.choice(difficulties, p=weights)

        # 确定动态参数
        if self.stage5_mode:
            actual_size = random.randint(10, self.max_size)
            num_fruit_types = random.randint(*self.cfg.stage5_fruit_range)
            min_match_length = self.cfg.stage5_match_length_range[0]
            max_match_length = random.randint(*self.cfg.stage5_match_length_range)
        elif self.random_size:
            actual_size = random.randint(10, self.max_size)
            num_fruit_types = None
            min_match_length = None
            max_match_length = None
        else:
            actual_size = self.size
            num_fruit_types = None
            min_match_length = None
            max_match_length = None

        board, mask, _ = self.generator.generate_board(
            actual_size, difficulty,
            num_fruit_types=num_fruit_types,
            min_match_length=min_match_length,
            max_match_length=max_match_length
        )

        # 若启用随机尺寸，将实际棋盘嵌入到 max_size x max_size 中随机位置
        if (self.random_size or self.stage5_mode) and actual_size < self.max_size:
            num_ft = num_fruit_types if num_fruit_types is not None else self.cfg.num_fruit_types
            full_board = self.generator.rng.randint(0, num_ft, (self.max_size, self.max_size))
            full_mask = np.zeros((self.max_size, self.max_size), dtype=np.bool_)
            sy = random.randint(0, self.max_size - actual_size)
            sx = random.randint(0, self.max_size - actual_size)
            full_board[sy:sy+actual_size, sx:sx+actual_size] = board
            full_mask[sy:sy+actual_size, sx:sx+actual_size] = mask
            board = full_board
            mask = full_mask
            actual_size = self.max_size

        fruit_ids = torch.from_numpy(board).long()

        # 正n边形顶点3通道编码: (max_fruit_types, H, W)
        # stage4/5 动态 n in 5-12 通过前 n 个通道表达，其余通道置零
        board_tensor = torch.zeros(self.cfg.max_fruit_types, board.shape[0], board.shape[1], dtype=torch.float32)
        board_t = torch.from_numpy(board).long().clamp(0, self.cfg.max_fruit_types - 1)
        board_tensor.scatter_(0, board_t.unsqueeze(0), 1.0)

        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return board_tensor, mask_tensor, fruit_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        return self._generate_one()
```

### 模块4: models/model.py — U-Net模型架构

```python
"""models/model.py - U-Net + 大卷积核 + 膨胀卷积 + 可选Mamba"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Match3Config


class ConvBlock(nn.Module):
    """基础卷积块: Conv -> BN -> ReLU"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, padding: int = None):
        super().__init__()
        if padding is None:
            padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """Encoder 块: 两个卷积 + 下采样"""

    def __init__(self, in_ch: int, out_ch: int, use_dilation: bool = False,
                 dilation_rate: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, 3, dilation_rate if use_dilation else 1)
        self.conv2 = ConvBlock(out_ch, out_ch, 3, dilation_rate if use_dilation else 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x  # 返回下采样后的特征 + skip connection


class DecoderBlock(nn.Module):
    """Decoder 块: 上采样 + 拼接 skip + 两个卷积"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = ConvBlock(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不匹配
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class Match3UNet(nn.Module):
    """三消 Pattern 识别 U-Net

    输入: (B, max_fruit_types+1, H, W) 正n边形顶点3通道 + valid_mask
    输出: (B, 1, H, W) sigmoid mask
    """

    def __init__(self, config: Match3Config):
        super().__init__()
        self.cfg = config

        # 首层: 大卷积核捕获局部连续性 (支持 正n边形顶点3通道输入)
        input_ch = getattr(config, 'max_fruit_types', config.num_fruit_types)
        self.stem = ConvBlock(input_ch, config.base_channels,
                              config.initial_kernel_size)

        # Encoder
        self.encoders = nn.ModuleList()
        ch = config.base_channels
        for i in range(config.num_encoder_blocks):
            next_ch = ch * 2
            dilation = config.dilation_rates[i] if config.use_dilation and i < len(config.dilation_rates) else 1
            self.encoders.append(EncoderBlock(ch, next_ch, config.use_dilation, dilation))
            ch = next_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(ch, ch * 2),
            ConvBlock(ch * 2, ch * 2)
        )
        ch = ch * 2

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(config.num_encoder_blocks - 1, -1, -1):
            skip_ch = config.base_channels * (2 ** i)
            out_ch = skip_ch
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch

        # 输出头: 1x1 卷积输出 mask
        self.final_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        # x: (B, fruit_embed_dim, H, W)
        x = self.stem(x)

        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return torch.sigmoid(self.final_conv(x))  # (B, 1, H, W)
```

### 模块5: trainer/losses.py — 组合损失函数

```python
"""trainer/losses.py - Dice + Focal + Boundary 组合损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import Match3Config


class DiceLoss(nn.Module):
    """Dice Loss for imbalanced segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class FocalLoss(nn.Module):
    """Focal Loss for hard examples"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BoundaryLoss(nn.Module):
    """边界损失: 惩罚 mask 边缘误差"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 使用拉普拉斯算子检测边界
        laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                       dtype=torch.float32, device=pred.device)
        laplace_kernel = laplace_kernel.view(1, 1, 3, 3)

        pred_boundary = F.conv2d(pred, laplace_kernel, padding=1).abs()
        target_boundary = F.conv2d(target.unsqueeze(1), laplace_kernel, padding=1).abs()

        return F.mse_loss(pred_boundary, target_boundary)


class Match3Loss(nn.Module):
    """组合损失函数: Dice + Focal + Boundary

    返回dict: {total, dice, focal, boundary}
    """

    def __init__(self, config: Match3Config):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(config.focal_alpha, config.focal_gamma)
        self.boundary = BoundaryLoss()
        self.w_dice = config.dice_weight
        self.w_focal = config.focal_weight
        self.w_boundary = config.boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        boundary_loss = self.boundary(pred, target)

        total = self.w_dice * dice_loss + self.w_focal * focal_loss + self.w_boundary * boundary_loss

        return {
            "total": total,
            "dice": dice_loss,
            "focal": focal_loss,
            "boundary": boundary_loss
        }
```

### 模块6: models/postprocess.py — 后处理规则约束层

```python
"""models/postprocess.py - 强制 mask 满足三消规则"""
import torch
import numpy as np
from typing import Tuple


def enforce_match3_rules(pred_mask: torch.Tensor, board: torch.Tensor,
                         min_length: int = 3, threshold: float = 0.5,
                         fruit_ids: torch.Tensor = None) -> torch.Tensor:
    """
    后处理: 强制 mask 满足三消规则

    流程:
    1. BFS连通分量检测
    2. 直线连续性验证 (只保留横向或纵向连续的连通区域)
    3. 只保留满足三消规则的连通区域 (长度>=3的直线)

    Args:
        pred_mask: (B, 1, H, W) 概率图
        board: (B, C, H, W) 棋盘状态 (one-hot 或 RoPE)
        min_length: 最小消除长度
        threshold: 二值化阈值
        fruit_ids: (B, H, W) LongTensor, 可选，直接传入fruit类型ID (RoPE模式下使用)

    Returns:
        corrected_mask: (B, 1, H, W) 修正后的概率 mask
    """
    B, _, H, W = pred_mask.shape
    binary_mask = (pred_mask > threshold).squeeze(1).cpu().numpy()  # (B, H, W)

    # 恢复颜色索引
    if fruit_ids is not None:
        color_board = fruit_ids.cpu().numpy()  # (B, H, W)
    else:
        color_board = torch.argmax(board, dim=1).cpu().numpy()  # (B, H, W)

    corrected = np.zeros_like(binary_mask, dtype=np.float32)

    for b in range(B):
        mask = binary_mask[b]
        colors = color_board[b]

        # 找到所有连通分量
        visited = np.zeros((H, W), dtype=np.bool_)

        for y in range(H):
            for x in range(W):
                if mask[y, x] and not visited[y, x]:
                    # BFS 找连通分量
                    component = []
                    queue = [(y, x)]
                    visited[y, x] = True
                    color = colors[y, x]

                    while queue:
                        cy, cx = queue.pop(0)
                        component.append((cy, cx))

                        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and \
                               not visited[ny, nx] and mask[ny, nx] and colors[ny, nx] == color:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                    # 检查是否满足三消规则 (直线型, 长度>=3)
                    if len(component) >= min_length:
                        ys, xs = zip(*component)
                        # 检查是否横向或纵向连续
                        is_horizontal = len(set(ys)) == 1 and max(xs) - min(xs) + 1 == len(component)
                        is_vertical = len(set(xs)) == 1 and max(ys) - min(ys) + 1 == len(component)

                        if is_horizontal or is_vertical:
                            for cy, cx in component:
                                corrected[b, cy, cx] = 1

    return torch.from_numpy(corrected).unsqueeze(1).to(pred_mask.device)
```

### 模块7: trainer/trainer.py — 训练引擎（含课程学习）

```python
"""trainer/trainer.py - 三消模型训练器 (含课程学习)"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict

from config import Match3Config
from data.data_generator import Match3Dataset
from models.model import Match3UNet
from trainer.losses import Match3Loss
from models.postprocess import enforce_match3_rules


class Match3Trainer:
    """三消模型训练器

    支持课程学习（5 stage递进）:
    - Stage 1: 10x10,  固定6种fruit, match 3-5, epochs 1-20
    - Stage 2: 25x25,  固定6种fruit, match 3-5, epochs 21-40
    - Stage 3: 50x50,  固定6种fruit, match 3-5, epochs 41-60
    - Stage 4: 随机10~50, 固定6种fruit, match 3-5, epochs 61-80
    - Stage 5: 随机10~50, 随机5~12种fruit, match 5-8, epochs 81-100
    """

    def __init__(self, config: Match3Config):
        self.cfg = config
        self.device = self._get_device()

        # 模型
        self.model = Match3UNet(config).to(self.device)

        # 损失
        self.criterion = Match3Loss(config)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度
        self.scheduler = self._build_scheduler()

        # 日志
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)

        self.global_step = 0

    def _get_device(self):
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.cfg.device)

    def _build_scheduler(self):
        if self.cfg.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.cfg.warmup_epochs, T_mult=2
            )
        elif self.cfg.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = {k: 0.0 for k in ["total", "dice", "focal", "boundary"]}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            boards = batch[0].to(self.device)
            masks = batch[1].to(self.device).unsqueeze(1)  # (B, 1, H, W)

            self.optimizer.zero_grad()
            preds = self.model(boards)
            losses = self.criterion(preds, masks)
            losses["total"].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            for k in total_loss:
                total_loss[k] += losses[k].item()

            # 日志
            if batch_idx % 10 == 0:
                self.writer.add_scalar("train/total_loss", losses["total"].item(), self.global_step)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

            self.global_step += 1

        return {k: v / len(dataloader) for k, v in total_loss.items()}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                boards = batch[0].to(self.device)
                masks = batch[1].to(self.device).unsqueeze(1)
                fruit_ids = batch[2].to(self.device) if len(batch) > 2 else None

                preds = self.model(boards)
                if self.cfg.enforce_connectivity:
                    preds = enforce_match3_rules(preds, boards, fruit_ids=fruit_ids)

                binary_preds = (preds > self.cfg.mask_threshold).float()

                # 计算指标
                intersection = (binary_preds * masks).sum(dim=(1,2,3))
                union = ((binary_preds + masks) > 0).float().sum(dim=(1,2,3))
                iou = (intersection / (union + 1e-6)).mean().item()

                precision = (intersection / (binary_preds.sum(dim=(1,2,3)) + 1e-6)).mean().item()
                recall = (intersection / (masks.sum(dim=(1,2,3)) + 1e-6)).mean().item()

                total_iou += iou
                total_precision += precision
                total_recall += recall

        n = len(dataloader)
        return {
            "iou": total_iou / n,
            "precision": total_precision / n,
            "recall": total_recall / n,
            "f1": 2 * (total_precision / n) * (total_recall / n) /
                  ((total_precision / n) + (total_recall / n) + 1e-6)
        }

    def fit(self, start_stage: int = 0):
        """完整训练流程 (含课程学习)"""
        stages = self.cfg.curriculum_stages if self.cfg.curriculum_enabled else [self.cfg.board_size]

        for stage_idx, stage_size in enumerate(stages):
            if stage_idx < start_stage:
                print(f"\n跳过阶段 {stage_idx} (尺寸 {stage_size}x{stage_size})")
                continue
            if stage_size == -2:
                stage_label = "Stage5: 随机10~50 + 动态fruit + match>=5"
            elif stage_size == -1:
                stage_label = "Stage4: 随机尺寸 10~50"
            else:
                stage_label = f"{stage_size}x{stage_size}"
            print(f"\n{':'*50}")
            print(f"课程学习阶段 {stage_idx + 1}/{len(stages)}: 棋盘尺寸 {stage_label}")
            print(f"{':'*50}")

            # 创建该阶段数据集
            train_ds = Match3Dataset(self.cfg, self.cfg.train_samples, stage_size)
            val_ds = Match3Dataset(self.cfg, self.cfg.val_samples, stage_size)

            train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                      shuffle=True, num_workers=self.cfg.num_workers)
            val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size,
                                    shuffle=False, num_workers=self.cfg.num_workers)

            epochs_this_stage = self.cfg.curriculum_epochs_per_stage if self.cfg.curriculum_enabled else self.cfg.num_epochs

            for epoch in range(1, epochs_this_stage + 1):
                global_epoch = sum(self.cfg.curriculum_epochs_per_stage for _ in range(stage_idx)) + epoch

                train_loss = self.train_epoch(train_loader, global_epoch)
                val_metrics = self.validate(val_loader)

                # 日志
                for k, v in train_loss.items():
                    self.writer.add_scalar(f"stage{stage_idx}/train_{k}", v, global_epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"stage{stage_idx}/val_{k}", v, global_epoch)

                print(f"Epoch {global_epoch}: Loss={train_loss['total']:.4f}, "
                      f"Val IoU={val_metrics['iou']:.4f}, F1={val_metrics['f1']:.4f}")

                # 学习率调度
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss["total"])
                else:
                    self.scheduler.step()

                # 保存检查点
                if global_epoch % self.cfg.save_every == 0:
                    self.save_checkpoint(global_epoch)

        print("\n训练完成!")
        self.save_checkpoint("final")

    def save_checkpoint(self, epoch):
        path = os.path.join(self.cfg.checkpoint_dir, f"match3_unet_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
```

### 模块8: trainer/inference.py — 推理与评估接口

```python
"""trainer/inference.py - 推理封装"""
import torch
import numpy as np
from typing import List, Tuple, Dict

from config import Match3Config
from models.model import Match3UNet
from data.data_generator import Match3BoardGenerator
from models.postprocess import enforce_match3_rules


class Match3Predictor:
    """推理封装"""

    def __init__(self, model_path: str, config: Match3Config = None, device: str = "auto"):
        if config is None:
            # 从检查点加载配置
            checkpoint = torch.load(model_path, map_location="cpu")
            config = checkpoint["config"]

        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)

        self.model = Match3UNet(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, board: np.ndarray) -> np.ndarray:
        """
        单张棋盘预测

        Args:
            board: (H, W) int array, 水果类型索引

        Returns:
            mask: (H, W) bool array, 消除位置
        """
        # 转 one-hot
        h, w = board.shape
        board_tensor = torch.zeros(self.cfg.num_fruit_types, h, w)
        for c in range(self.cfg.num_fruit_types):
            board_tensor[c] = torch.from_numpy(board == c)

        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            pred = self.model(board_tensor)
            if self.cfg.enforce_connectivity:
                pred = enforce_match3_rules(pred, board_tensor)

        mask = (pred.squeeze().cpu().numpy() > self.cfg.mask_threshold)
        return mask

    def batch_predict(self, boards: List[np.ndarray]) -> List[np.ndarray]:
        """批量预测

        Args:
            boards: List of (H, W) int arrays

        Returns:
            List of (H, W) bool arrays
        """
        # 假设所有 boards 尺寸相同
        h, w = boards[0].shape
        batch = torch.zeros(len(boards), self.cfg.num_fruit_types, h, w)

        for i, board in enumerate(boards):
            for c in range(self.cfg.num_fruit_types):
                batch[i, c] = torch.from_numpy(board == c)

        batch = batch.to(self.device)

        with torch.no_grad():
            preds = self.model(batch)
            if self.cfg.enforce_connectivity:
                preds = enforce_match3_rules(preds, batch)

        masks = [(p.squeeze().cpu().numpy() > self.cfg.mask_threshold) for p in preds]
        return masks

    def evaluate_accuracy(self, num_test_samples: int = 1000) -> Dict[str, float]:
        """在随机数据上评估模型 vs 规则引擎的准确率"""
        generator = Match3BoardGenerator(self.cfg)
        correct = 0
        total = 0

        for _ in range(num_test_samples):
            board, true_mask = generator.generate_board(self.cfg.board_size, "positive")
            pred_mask = self.predict(board)

            correct += (pred_mask == true_mask).sum()
            total += true_mask.size

        return {
            "pixel_accuracy": correct / total,
            "test_samples": num_test_samples
        }
```

### 模块9: main.py — CLI主入口

```python
"""main.py - CLI入口"""
import argparse
import json
import numpy as np

from config import Match3Config
from trainer.trainer import Match3Trainer
from trainer.inference import Match3Predictor


def main():
    parser = argparse.ArgumentParser(description="Match-3 Pattern Recognition Trainer")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--config", type=str, help="JSON config file path")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint for eval/infer")
    parser.add_argument("--board", type=str, help="Numpy board file for inference")
    args = parser.parse_args()

    # 加载配置
    config = Match3Config()
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    if args.mode == "train":
        trainer = Match3Trainer(config)
        trainer.fit()

    elif args.mode == "eval":
        predictor = Match3Predictor(args.checkpoint, config)
        metrics = predictor.evaluate_accuracy()
        print(f"评估结果: {metrics}")

    elif args.mode == "infer":
        predictor = Match3Predictor(args.checkpoint, config)
        board = np.load(args.board)
        mask = predictor.predict(board)
        np.save("predicted_mask.npy", mask)
        print(f"预测 mask 已保存至 predicted_mask.npy")


if __name__ == "__main__":
    main()
```

---

## 使用示例

### 快速训练命令

```bash
# 1. 安装依赖
pip install torch>=2.0.0 torchvision numpy pillow tqdm tensorboard albumentations opencv-python

# 2. 默认配置训练 (100x100 棋盘, 50k 样本)
python main.py --mode train

# 3. 自定义配置训练
python main.py --mode train --config custom_config.json
```

### 自定义配置JSON示例 (custom_config.json)

```json
{
    "board_size": 100,
    "num_fruit_types": 6,
    "train_samples": 100000,
    "batch_size": 80,
    "num_epochs": 150,
    "learning_rate": 0.001,
    "encoder_backbone": "custom_cnn",
    "curriculum_enabled": true,
    "curriculum_stages": [10, 25, 50, -1, -3, -2],
    "enforce_connectivity": true,
    "dice_weight": 0.5,
    "focal_weight": 0.3,
    "boundary_weight": 0.2
}
```

### Python推理示例

```python
from trainer.inference import Match3Predictor
import numpy as np

# 加载训练好的模型
predictor = Match3Predictor("./checkpoints/match3_unet_epochfinal.pt")

# 你的游戏棋盘 (100x100, 每个值是水果类型 0-5)
my_board = np.random.randint(0, 6, (100, 100))

# 预测消除 mask
elimination_mask = predictor.predict(my_board)
# elimination_mask: True 的位置就是可以消除的格点

# 获取消除坐标
elimination_coords = np.argwhere(elimination_mask)
print(f"检测到 {len(elimination_coords)} 个可消除格点")
```

---

## 性能预期与调优建议

| 棋盘尺寸 | 预期 Val IoU | 训练时间 (RTX 4090) | 显存占用 |
|----------|-------------|---------------------|----------|
| 20x20    | > 0.95      | ~10 min             | ~2 GB    |
| 50x50    | > 0.90      | ~1 hour             | ~4 GB    |
| 100x100  | > 0.85      | ~4 hours            | ~8 GB    |

**如果 IoU 不达标**:
1. 增加 `train_samples` 到 200k+
2. 提高 `hard` 样本比例到 0.5
3. 尝试 `focal_gamma` = 3.0 (更聚焦难例)
4. 增加 `num_encoder_blocks` 到 5-6
5. 使用 `boundary_weight` = 0.3 强化边缘精度

---

## 快捷工作流（常用脚本）

> 当用户说"帮我push"时，按以下顺序执行：

### 1. 导出 TensorBoard 训练数据

```bash
python scripts/export_tensorboard.py
```

输出：
- `logs/training_loss.json` — 结构化 scalar 数据
- `logs/TRAINING_LOG.md` — 人类可读 Markdown 报告（损失用科学计数法）

### 2. 绘制训练曲线（处理跨 stage 数量级差异）

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

### 3. Git 提交并 Push

```bash
# 确保 .gitignore 允许日志产物
git add .gitignore scripts/export_tensorboard.py scripts/plot_training_curves.py \
    logs/training_loss.json logs/TRAINING_LOG.md logs/training_curves/

git commit -m "docs: update training logs and curves"
git push origin main
```

> **注意**：只 add 与日志/可视化相关的文件，不提交训练中间产物（checkpoints/大量 visualization PNG）。

---

## 文件引用标签

- `match3-pattern-core` - Pattern类型核心定义与检测算法
- `match3-data-generation` - 三消棋盘数据生成引擎
- `match3-cnn-unet-trainer` - U-Net模型训练pipeline
- `match3-training-foundation` - 训练基础设施 (损失/优化/调度)
- `match3-shared-utils` - 共享工具函数
