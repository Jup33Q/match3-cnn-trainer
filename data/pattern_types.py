"""pattern_types.py - Pattern类型枚举 + PatternDetector完整实现"""
from enum import Enum
from typing import Tuple, List, Dict
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
        - 总共5格 (3+3-1)
        """
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
        - 横向3连 + 纵向3连，共享中心格点且该中心为某条线的端点或中点
        """
        patterns = []
        h3_list = [p for p in line_patterns if p["type"] == PatternType.H3]
        v3_list = [p for p in line_patterns if p["type"] == PatternType.V3]

        for h3 in h3_list:
            h3_coords = set(h3["coords"])
            h3_color = h3["color"]
            h3_xs = sorted([c[1] for c in h3["coords"]])

            for v3 in v3_list:
                if v3["color"] != h3_color:
                    continue

                v3_coords = set(v3["coords"])
                intersection = h3_coords & v3_coords

                if len(intersection) == 1:
                    center = list(intersection)[0]
                    all_coords = list(h3_coords | v3_coords)

                    if len(all_coords) == 5:
                        v3_ys = sorted([c[0] for c in v3["coords"]])

                        # T型: 中心是H3的中点，且是V3的端点（或反之）
                        is_h_center = (center[1] == h3_xs[1])  # H3中点
                        is_v_end = (center[0] == v3_ys[0] or center[0] == v3_ys[2])  # V3端点
                        is_v_center = (center[0] == v3_ys[1])  # V3中点
                        is_h_end = (center[1] == h3_xs[0] or center[1] == h3_xs[2])  # H3端点

                        if (is_h_center and is_v_end) or (is_v_center and is_h_end):
                            patterns.append({
                                "type": PatternType.T,
                                "type_id": PatternType.T.value,
                                "color": h3_color,
                                "coords": all_coords,
                                "priority": PatternDetector.PATTERN_PRIORITY[PatternType.T],
                                "center": center
                            })

        return patterns

    @staticmethod
    def _detect_crosses(board: np.ndarray, line_patterns: List[Dict]) -> List[Dict]:
        """
        检测十字型Pattern:
        - 横向3连 + 纵向3连，共享中心格点，且中心是两条线的中点
        """
        patterns = []
        h3_list = [p for p in line_patterns if p["type"] == PatternType.H3]
        v3_list = [p for p in line_patterns if p["type"] == PatternType.V3]

        for h3 in h3_list:
            h3_coords = set(h3["coords"])
            h3_color = h3["color"]
            h3_xs = sorted([c[1] for c in h3["coords"]])

            for v3 in v3_list:
                if v3["color"] != h3_color:
                    continue

                v3_coords = set(v3["coords"])
                v3_ys = sorted([c[0] for c in v3["coords"]])

                intersection = h3_coords & v3_coords

                if len(intersection) == 1:
                    center = list(intersection)[0]
                    all_coords = list(h3_coords | v3_coords)

                    if len(all_coords) == 5:
                        # 检查中心是否是两条线的中点
                        is_h_center = center[1] == h3_xs[1]
                        is_v_center = center[0] == v3_ys[1]

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


def pattern_priority_resolve(candidates):
    """根据优先级从多个候选Pattern中选择最高优先级的类型"""
    if not candidates:
        return PatternType.NONE
    return max(candidates, key=lambda p: PatternDetector.PATTERN_PRIORITY.get(p, 0))


def is_valid_pattern(board: np.ndarray, positions):
    """验证给定位置是否构成有效的同类型连通结构"""
    if len(positions) < 3:
        return False
    values = [board[y, x] for y, x in positions]
    return len(set(values)) == 1
