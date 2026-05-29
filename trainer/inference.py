"""inference.py - 推理封装"""
import torch
import numpy as np
from typing import List, Dict

from config import Match3Config
from models.model import Match3UNet
from data.data_generator import Match3BoardGenerator
from models.postprocess import enforce_match3_rules


class Match3Predictor:
    """推理封装"""

    def __init__(self, model_path: str, config: Match3Config = None, device: str = "auto"):
        if config is None:
            # 从检查点加载配置
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]

        self.cfg = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = Match3UNet(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _encode_board(self, board: np.ndarray) -> torch.Tensor:
        """将 numpy board 编码为模型输入（正n边形顶点3通道 + 可选 valid_mask）"""
        h, w = board.shape
        n_colors = self.cfg.num_fruit_types

        # 正 n 边形顶点 3 通道编码 (BF16 quantize)
        angles = 2 * np.pi * board / n_colors
        board_np = np.stack([
            np.cos(angles),
            np.sin(angles),
            np.ones_like(angles)
        ], axis=0).astype(np.float32)  # (3, H, W)
        board_tensor = torch.from_numpy(board_np).to(torch.bfloat16)

        if getattr(self.cfg, 'use_valid_mask', True):
            valid_mask = torch.ones(1, h, w, dtype=torch.bfloat16)
            board_tensor = torch.cat([board_tensor, valid_mask], dim=0)  # (4, H, W)
        return board_tensor

    def predict(self, board: np.ndarray) -> np.ndarray:
        """
        单张棋盘预测

        Args:
            board: (H, W) int array, 水果类型索引

        Returns:
            mask: (H, W) bool array, 消除位置
        """
        board_tensor = self._encode_board(board).unsqueeze(0).to(self.device)  # (1, C, H, W)
        fruit_ids = torch.from_numpy(board).long().unsqueeze(0).to(self.device)  # (1, H, W)

        # 根据配置启用 BF16/FP16 autocast，与训练时保持一致
        autocast_dtype = None
        if self.cfg.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif self.cfg.precision == "fp16" and torch.cuda.is_available():
            autocast_dtype = torch.float16

        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    pred = self.model(board_tensor)
                    pred = torch.sigmoid(pred)
                    if self.cfg.enforce_connectivity:
                        pred = enforce_match3_rules(pred, board_tensor, fruit_ids=fruit_ids)
            else:
                pred = self.model(board_tensor)
                pred = torch.sigmoid(pred)
                if self.cfg.enforce_connectivity:
                    pred = enforce_match3_rules(pred, board_tensor, fruit_ids=fruit_ids)

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
        batch_list = [self._encode_board(b) for b in boards]
        batch = torch.stack(batch_list, dim=0).to(self.device)  # (B, C, H, W)
        fruit_ids = torch.stack([torch.from_numpy(b).long() for b in boards], dim=0).to(self.device)

        # 根据配置启用 BF16/FP16 autocast，与训练时保持一致
        autocast_dtype = None
        if self.cfg.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif self.cfg.precision == "fp16" and torch.cuda.is_available():
            autocast_dtype = torch.float16

        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    preds = self.model(batch)
                    preds = torch.sigmoid(preds)
                    if self.cfg.enforce_connectivity:
                        preds = enforce_match3_rules(preds, batch, fruit_ids=fruit_ids)
            else:
                preds = self.model(batch)
                preds = torch.sigmoid(preds)
                if self.cfg.enforce_connectivity:
                    preds = enforce_match3_rules(preds, batch, fruit_ids=fruit_ids)

        masks = [(p.squeeze().cpu().numpy() > self.cfg.mask_threshold) for p in preds]
        return masks

    def evaluate_accuracy(self, num_test_samples: int = 1000) -> Dict[str, float]:
        """在随机数据上评估模型 vs 规则引擎的准确率"""
        generator = Match3BoardGenerator(self.cfg)
        correct = 0
        total = 0
        total_iou = 0.0

        for _ in range(num_test_samples):
            board, true_mask, _ = generator.generate_board(self.cfg.board_size, difficulty="positive")
            pred_mask = self.predict(board)

            correct += (pred_mask == true_mask).sum()
            total += true_mask.size

            intersection = (pred_mask & true_mask).sum()
            union = (pred_mask | true_mask).sum()
            total_iou += intersection / (union + 1e-6)

        return {
            "pixel_accuracy": correct / total,
            "mean_iou": total_iou / num_test_samples,
            "test_samples": num_test_samples
        }
