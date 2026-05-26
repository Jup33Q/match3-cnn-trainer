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
        board_tensor = torch.zeros(self.cfg.num_fruit_types, h, w, dtype=torch.float32)
        for c in range(self.cfg.num_fruit_types):
            board_tensor[c] = torch.from_numpy((board == c).astype(np.float32))

        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            pred = self.model(board_tensor)
            pred = torch.sigmoid(pred)
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
        batch = torch.zeros(len(boards), self.cfg.num_fruit_types, h, w, dtype=torch.float32)

        for i, board in enumerate(boards):
            for c in range(self.cfg.num_fruit_types):
                batch[i, c] = torch.from_numpy((board == c).astype(np.float32))

        batch = batch.to(self.device)

        with torch.no_grad():
            preds = self.model(batch)
            preds = torch.sigmoid(preds)
            if self.cfg.enforce_connectivity:
                preds = enforce_match3_rules(preds, batch)

        masks = [(p.squeeze().cpu().numpy() > self.cfg.mask_threshold) for p in preds]
        return masks

    def evaluate_accuracy(self, num_test_samples: int = 1000) -> Dict[str, float]:
        """在随机数据上评估模型 vs 规则引擎的准确率"""
        generator = Match3BoardGenerator(self.cfg)
        correct = 0
        total = 0
        total_iou = 0.0

        for _ in range(num_test_samples):
            board, true_mask, _ = generator.generate_board(self.cfg.board_size, "positive")
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
