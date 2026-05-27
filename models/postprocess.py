"""postprocess.py - 强制 mask 满足三消规则"""
import torch
import numpy as np


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

                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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
                                corrected[b, cy, cx] = 1.0

    return torch.from_numpy(corrected).unsqueeze(1).to(pred_mask.device)
