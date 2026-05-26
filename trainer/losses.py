"""losses.py - Dice + Focal + Boundary 组合损失 (Logits输入，BF16兼容)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import Match3Config


class DiceLoss(nn.Module):
    """Dice Loss for imbalanced segmentation (输入logits)"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class FocalLoss(nn.Module):
    """Focal Loss for hard examples (输入logits，BF16安全)"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 使用 bce_with_logits，兼容 BF16 autocast
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BoundaryLoss(nn.Module):
    """边界损失: 惩罚 mask 边缘误差 (输入logits)"""

    def __init__(self):
        super().__init__()
        laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                       dtype=torch.float32)
        self.register_buffer("laplace_kernel", laplace_kernel.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 确保kernel与输入同设备同类型
        kernel = self.laplace_kernel.to(device=pred.device, dtype=pred.dtype)
        pred_boundary = F.conv2d(pred, kernel, padding=1).abs()
        target_boundary = F.conv2d(target, kernel, padding=1).abs()

        return F.mse_loss(pred_boundary, target_boundary)


class Match3Loss(nn.Module):
    """组合损失函数: Dice + Focal + Boundary

    输入: logits (B, 1, H, W)
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
        if target.dim() == 3:
            target = target.unsqueeze(1)

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
