"""data_generator.py - 程序化生成三消棋盘与消除Mask"""
import torch
import numpy as np
from typing import Tuple

from config import Match3Config
from .pattern_types import PatternDetector, PatternType


class Match3BoardGenerator:
    """程序化生成三消棋盘与消除 Mask"""

    def __init__(self, config: Match3Config):
        self.cfg = config
        self.rng = np.random.RandomState(config.seed)
        self.detector = PatternDetector()

    def generate_board(self, size: int, difficulty: str = "mixed") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成棋盘和对应的三消 Mask

        Args:
            size: 棋盘尺寸 (size x size)
            difficulty: "easy" | "hard" | "positive"

        Returns:
            board: (size, size) int array, 每个格子的水果类型 [0, num_fruit_types)
            mask: (size, size) bool array, 1表示该格点参与三消
            pattern_types: (size, size) int array, Pattern类型ID
        """
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
        self.size = stage_size or config.board_size
        self.generator = Match3BoardGenerator(config)

        # 预生成所有数据 (内存允许时) 或 动态生成
        self.preload = num_samples <= 10000
        if self.preload:
            self.data = [self._generate_one() for _ in range(num_samples)]

    def _generate_one(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成单个样本"""
        # 按难度比例采样
        difficulties = ["easy", "hard", "positive"]
        weights = self.cfg.difficulty_ratio
        difficulty = np.random.choice(difficulties, p=weights)

        board, mask, _ = self.generator.generate_board(self.size, difficulty)

        # 转换为 one-hot 张量: (C, H, W)
        board_tensor = torch.zeros(self.cfg.num_fruit_types, self.size, self.size, dtype=torch.float32)
        for c in range(self.cfg.num_fruit_types):
            board_tensor[c] = torch.from_numpy((board == c).astype(np.float32))

        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return board_tensor, mask_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        return self._generate_one()
