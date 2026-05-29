"""data_generator.py - 程序化生成三消棋盘与消除Mask"""
import random
import torch
import numpy as np
from typing import Tuple

from config import Match3Config
from .pattern_types import PatternDetector, PatternType


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

    def generate_board(self, height: int, width: int = None, difficulty: str = "mixed",
                       num_fruit_types: int = None,
                       min_match_length: int = None,
                       max_match_length: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成棋盘和对应的三消 Mask

        Args:
            height: 棋盘高度
            width: 棋盘宽度 (None=height，即正方形)
            difficulty: "easy" | "hard" | "positive"
            num_fruit_types: 临时覆盖的fruit种类数 (None=使用config)
            min_match_length: 临时覆盖的最小消除长度 (None=使用config)
            max_match_length: 临时覆盖的最大注入长度 (None=使用config)

        Returns:
            board: (height, width) int array, 每个格子的水果类型 [0, num_fruit_types)
            mask: (height, width) bool array, 1表示该格点参与三消
            pattern_types: (height, width) int array, Pattern类型ID
        """
        if width is None:
            width = height

        # 保存旧值以便恢复
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
                board = self.rng.randint(0, self.cfg.num_fruit_types, (height, width))
                pattern_map, patterns = self.detector.detect_all_patterns(board, self.cfg.min_match_length)
                mask = (pattern_map != PatternType.NONE.value)

            elif difficulty == "hard":
                board = self._generate_near_miss_board(height, width)
                mask = np.zeros((height, width), dtype=np.bool_)
                pattern_map = np.full((height, width), PatternType.NONE.value, dtype=np.int32)

            else:  # "positive"
                board = self.rng.randint(0, self.cfg.num_fruit_types, (height, width))
                board, mask, pattern_map = self._inject_match_patterns(board)

            return board, mask, pattern_map
        finally:
            self.cfg.num_fruit_types = old_num
            self.cfg.min_match_length = old_min
            self.cfg.max_match_length = old_max

    def _generate_near_miss_board(self, height: int, width: int = None) -> np.ndarray:
        """生成含大量接近三消但无实际消除的棋盘"""
        if width is None:
            width = height
        board = self.rng.randint(0, self.cfg.num_fruit_types, (height, width))

        # 注入两连结构
        num_patterns = max(height, width) * 2
        for _ in range(num_patterns):
            y, x = self.rng.randint(0, height, 1)[0], self.rng.randint(0, width, 1)[0]
            color = self.rng.randint(0, self.cfg.num_fruit_types)

            if self.rng.random() < 0.5 and x + 1 < width:
                board[y, x] = color
                board[y, x+1] = color
                if x + 2 < width:
                    board[y, x+2] = (color + 1) % self.cfg.num_fruit_types
                if x - 1 >= 0:
                    board[y, x-1] = (color + 2) % self.cfg.num_fruit_types
            elif y + 1 < height:
                board[y, x] = color
                board[y+1, x] = color
                if y + 2 < height:
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
        # 选择角点位置，确保有足够空间
        corner_y = self.rng.randint(1, h - 1)
        corner_x = self.rng.randint(1, w - 1)

        # 随机选择方向
        h_dir = self.rng.choice([-1, 1])  # 水平延伸方向
        v_dir = self.rng.choice([-1, 1])  # 垂直延伸方向

        # 水平方向: 角点 + 2个延伸
        hx_start = min(corner_x, corner_x + h_dir * 2)
        hx_end = max(corner_x, corner_x + h_dir * 2)
        if 0 <= hx_start < hx_end < w:
            board[corner_y, hx_start:hx_end+1] = color

        # 垂直方向: 角点 + 2个延伸
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

        # 横向3
        hx_start = max(0, center_x - 1)
        hx_end = min(w - 1, center_x + 1)
        board[center_y, hx_start:hx_end+1] = color

        # 纵向: 从中心向下（或向上）延伸2格
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

        # 横向3
        hx_start = max(0, center_x - 1)
        hx_end = min(w - 1, center_x + 1)
        board[center_y, hx_start:hx_end+1] = color

        # 纵向3
        vy_start = max(0, center_y - 1)
        vy_end = min(h - 1, center_y + 1)
        board[vy_start:vy_end+1, center_x] = color


class Match3Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset 封装，支持预加载"""

    def __init__(self, config: Match3Config, num_samples: int, stage_size: int = None):
        self.cfg = config
        self.num_samples = num_samples
        self.size = stage_size if stage_size is not None else config.board_size
        self.random_size = (stage_size == -1)          # 正方形随机 10~50（过渡阶段）
        self.rectangular_size = (stage_size == -3)     # 长方形随机（宽高独立 10~50）
        self.stage5_mode = (stage_size == -2)          # 长方形随机 + 颜色随机
        self.max_size = config.board_size
        self.generator = Match3BoardGenerator(config)

        # 预生成所有数据 (内存允许时) 或 动态生成
        self.preload = num_samples <= 10000 and not self.random_size and not self.rectangular_size and not self.stage5_mode
        if self.preload:
            self.data = [self._generate_one() for _ in range(num_samples)]

    def _generate_one(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成单个样本，返回 (board_onehot, mask, fruit_ids, valid_mask)"""
        # 按难度比例采样
        difficulties = ["easy", "hard", "positive"]
        weights = self.cfg.difficulty_ratio
        difficulty = np.random.choice(difficulties, p=weights)

        # 确定动态参数
        if self.stage5_mode:
            actual_h = random.randint(10, self.max_size)
            actual_w = random.randint(10, self.max_size)
            num_fruit_types = random.randint(*self.cfg.stage5_fruit_range)
            min_match_length = self.cfg.stage5_match_length_range[0]
            max_match_length = random.randint(*self.cfg.stage5_match_length_range)
        elif self.rectangular_size:
            actual_h = random.randint(10, self.max_size)
            actual_w = random.randint(10, self.max_size)
            num_fruit_types = None
            min_match_length = None
            max_match_length = None
        elif self.random_size:
            actual_h = actual_w = random.randint(10, self.max_size)
            num_fruit_types = None
            min_match_length = None
            max_match_length = None
        else:
            actual_h = actual_w = self.size
            num_fruit_types = None
            min_match_length = None
            max_match_length = None

        board, mask, _ = self.generator.generate_board(
            actual_h, actual_w, difficulty,
            num_fruit_types=num_fruit_types,
            min_match_length=min_match_length,
            max_match_length=max_match_length
        )

        valid_mask = np.ones((actual_h, actual_w), dtype=np.float32)

        # 若启用随机尺寸，将实际棋盘嵌入到 max_size x max_size 中随机位置
        needs_embed = (self.random_size or self.rectangular_size or self.stage5_mode)
        if needs_embed and (actual_h < self.max_size or actual_w < self.max_size):
            num_ft = num_fruit_types if num_fruit_types is not None else self.cfg.num_fruit_types
            full_board = self.generator.rng.randint(0, num_ft, (self.max_size, self.max_size))
            full_mask = np.zeros((self.max_size, self.max_size), dtype=np.bool_)
            full_valid = np.zeros((self.max_size, self.max_size), dtype=np.float32)
            sy = random.randint(0, self.max_size - actual_h)
            sx = random.randint(0, self.max_size - actual_w)
            full_board[sy:sy+actual_h, sx:sx+actual_w] = board
            full_mask[sy:sy+actual_h, sx:sx+actual_w] = mask
            full_valid[sy:sy+actual_h, sx:sx+actual_w] = 1.0
            board = full_board
            mask = full_mask
            valid_mask = full_valid
            actual_h = actual_w = self.max_size

        fruit_ids = torch.from_numpy(board).long()

        # 正 n 边形顶点 3 通道编码: (3, H, W)
        # n = 当前样本颜色种类数, m = fruit ID (0~n-1)
        # ch0 = cos(2π·m/n), ch1 = sin(2π·m/n), ch2 = 1
        # sin/cos 值做 BF16 quantize，与训练精度保持一致
        n_colors = num_fruit_types if num_fruit_types is not None else self.cfg.num_fruit_types
        angles = 2 * np.pi * board / n_colors
        board_np = np.stack([
            np.cos(angles),
            np.sin(angles),
            np.ones_like(angles)
        ], axis=0).astype(np.float32)  # (3, H, W)
        board_tensor = torch.from_numpy(board_np).to(torch.bfloat16)

        # 若启用 valid_mask 通道，拼接到输入（让模型分辨实际棋盘区域 vs padding）
        if getattr(self.cfg, 'use_valid_mask', True):
            valid_ch = torch.from_numpy(valid_mask).unsqueeze(0).to(torch.bfloat16)  # (1, H, W)
            board_tensor = torch.cat([board_tensor, valid_ch], dim=0)  # (4, H, W)

        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        valid_tensor = torch.from_numpy(valid_mask)
        return board_tensor, mask_tensor, fruit_ids, valid_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        return self._generate_one()
