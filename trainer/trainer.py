"""trainer.py - BF16混合精度训练引擎 (含课程学习 + 显存监控 + 自动Git提交)"""
import os
import random
import subprocess
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict
from PIL import Image
import numpy as np

from config import Match3Config
from data.data_generator import Match3Dataset
from models.model import Match3UNet
from .losses import Match3Loss
from models.postprocess import enforce_match3_rules
from .memory_monitor import MemoryMonitor

# 水果颜色映射 (RGB)
FRUIT_COLORS = [
    (255, 100, 100),   # 0: 红
    (100, 255, 100),   # 1: 绿
    (100, 100, 255),   # 2: 蓝
    (255, 255, 100),   # 3: 黄
    (255, 100, 255),   # 4: 紫
    (100, 255, 255),   # 5: 青
]


class Match3Trainer:
    """三消模型训练器 (BF16优化版)

    支持:
    - BF16/FP16/FP32 混合精度训练
    - 课程学习（多stage递进棋盘尺寸）
    - 每N batch记录日志
    - 显存监控与6GB限制
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

        # 混合精度 (BF16/FP16)
        self.scaler = None
        self.autocast_dtype = None
        if config.precision == "bf16" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
                print("[INFO] 启用 BF16 混合精度训练")
            else:
                print("[WARN] 设备不支持BF16，回退到FP32")
        elif config.precision == "fp16" and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast_dtype = torch.float16
            print("[INFO] 启用 FP16 混合精度训练 (AMP)")
        else:
            print("[INFO] 使用 FP32 训练")

        # 日志
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)
        self.memory_monitor = MemoryMonitor(config, self.writer)

        self.global_step = 0
        self.best_iou = 0.0

    def _get_device(self):
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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

    def _autocast_context(self):
        """获取自动类型转换上下文"""
        if self.autocast_dtype is not None:
            return torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        # 返回一个虚拟上下文管理器
        from contextlib import nullcontext
        return nullcontext()

    def train_epoch(self, dataloader: DataLoader, epoch: int, stage_idx: int) -> Dict[str, float]:
        self.model.train()
        total_loss = {k: 0.0 for k in ["total", "dice", "focal", "boundary"]}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (boards, masks) in enumerate(pbar):
            boards = boards.to(self.device)
            masks = masks.to(self.device).unsqueeze(1)  # (B, 1, H, W)

            self.optimizer.zero_grad()

            with self._autocast_context():
                preds = self.model(boards)
                losses = self.criterion(preds, masks)

            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            for k in total_loss:
                total_loss[k] += losses[k].item()

            # 每N batch记录日志
            if batch_idx % self.cfg.save_every_batches == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train_step/{k}", v.item(), self.global_step)
                self.writer.add_scalar("train_step/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.memory_monitor.log_to_tensorboard(self.global_step, "train_memory")

                # 显存检查
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    if mem_allocated > self.cfg.max_memory_gb:
                        print(f"\n⚠️ 显存超限! 当前={mem_allocated:.2f}GB, 限制={self.cfg.max_memory_gb}GB")
                        torch.cuda.empty_cache()

            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            self.global_step += 1

        return {k: v / len(dataloader) for k, v in total_loss.items()}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0

        with torch.no_grad():
            for boards, masks in tqdm(dataloader, desc="Validating"):
                boards = boards.to(self.device)
                masks = masks.to(self.device).unsqueeze(1)

                with self._autocast_context():
                    preds = self.model(boards)
                    if self.cfg.enforce_connectivity:
                        preds = enforce_match3_rules(preds, boards)

                binary_preds = (torch.sigmoid(preds) > self.cfg.mask_threshold).float()

                # 计算指标
                intersection = (binary_preds * masks).sum(dim=(1, 2, 3))
                union = ((binary_preds + masks) > 0).float().sum(dim=(1, 2, 3))
                iou = (intersection / (union + 1e-6)).mean().item()

                precision = (intersection / (binary_preds.sum(dim=(1, 2, 3)) + 1e-6)).mean().item()
                recall = (intersection / (masks.sum(dim=(1, 2, 3)) + 1e-6)).mean().item()

                total_iou += iou
                total_precision += precision
                total_recall += recall

        n = len(dataloader)
        avg_precision = total_precision / n
        avg_recall = total_recall / n
        return {
            "iou": total_iou / n,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-6)
        }

    def fit(self, start_stage: int = 0, start_epoch: int = 0):
        """完整训练流程 (含课程学习)

        Args:
            start_stage: 课程学习从第几个阶段开始 (0-based)
            start_epoch: 从哪个 global epoch 继续 (恢复训练时使用)
        """
        stages = self.cfg.curriculum_stages if self.cfg.curriculum_enabled else [self.cfg.board_size]

        print(f"\n{'='*60}")
        print(f"开始训练: 设备={self.device}")
        print(f"模型参数量={sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        print(f"精度模式={self.cfg.precision}")
        print(f"Batch Size={self.cfg.batch_size}")
        print(f"课程学习阶段={stages}")
        print(f"从第 {start_stage} 阶段开始")
        if start_epoch > 0:
            print(f"从 Epoch {start_epoch} 继续训练 (跳过已完成的 epoch)")
        print(f"{'='*60}\n")

        self.memory_monitor.print_summary()
        self.memory_monitor.reset_peak_stats()

        for stage_idx, stage_size in enumerate(stages):
            if stage_idx < start_stage:
                print(f"\n跳过阶段 {stage_idx} (尺寸 {stage_size}x{stage_size})")
                continue
            print(f"\n{':'*50}")
            print(f"课程学习阶段 {stage_idx + 1}/{len(stages)}: 棋盘尺寸 {stage_size}x{stage_size}")
            print(f"{':'*50}")

            # 创建该阶段数据集
            train_ds = Match3Dataset(self.cfg, self.cfg.train_samples, stage_size)
            val_ds = Match3Dataset(self.cfg, self.cfg.val_samples, stage_size)

            pin_memory = getattr(self.cfg, 'pin_memory', True)
            prefetch_factor = getattr(self.cfg, 'prefetch_factor', 2)
            persistent = self.cfg.num_workers > 0

            train_loader = DataLoader(
                train_ds, batch_size=self.cfg.batch_size,
                shuffle=True, num_workers=self.cfg.num_workers,
                pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                persistent_workers=persistent
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.cfg.batch_size,
                shuffle=False, num_workers=self.cfg.num_workers,
                pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                persistent_workers=persistent
            )

            epochs_this_stage = self.cfg.curriculum_epochs_per_stage if self.cfg.curriculum_enabled else self.cfg.num_epochs

            for epoch in range(1, epochs_this_stage + 1):
                global_epoch = sum(self.cfg.curriculum_epochs_per_stage for _ in range(stage_idx)) + epoch

                # 恢复训练时跳过已完成的 epoch，但保持 scheduler 状态同步
                if global_epoch <= start_epoch:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(0.0)
                    else:
                        self.scheduler.step()
                    continue

                train_loss = self.train_epoch(train_loader, global_epoch, stage_idx)
                val_metrics = self.validate(val_loader)

                # 每N个epoch生成可视化样例，减少CPU阻塞导致的GPU空闲
                vis_every = getattr(self.cfg, 'visualize_every_epochs', 1)
                if vis_every > 0 and global_epoch % vis_every == 0:
                    self.model.eval()
                    vis_samples = []
                    with torch.no_grad():
                        for vis_idx in range(10):
                            vis_board, vis_mask = None, None
                            for _ in range(50):  # 最多尝试50次生成正例 (减少CPU阻塞)
                                vb, vm = val_ds._generate_one()
                                if vm.sum() > 0:
                                    vis_board, vis_mask = vb, vm
                                    break
                            if vis_board is None:
                                vis_board, vis_mask = val_ds._generate_one()

                            vis_board = vis_board.unsqueeze(0).to(self.device)
                            vis_mask = vis_mask.unsqueeze(0).to(self.device)
                            with self._autocast_context():
                                vis_pred = self.model(vis_board)

                            sample_type = "positive" if vis_mask[0].sum() > 0 else "negative"
                            vis_samples.append((
                                vis_board[0], vis_mask[0], vis_pred[0], sample_type
                            ))
                        self._visualize_samples(vis_samples, global_epoch, stage_size)

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

                # 保存最佳模型
                if val_metrics["iou"] > self.best_iou:
                    self.best_iou = val_metrics["iou"]
                    self.save_checkpoint("best", stage_idx)
                    print(f"  *** 新最佳模型 (IoU={self.best_iou:.4f}) ***")

                # 每N epoch保存检查点
                if global_epoch % self.cfg.save_every_epochs == 0:
                    self.save_checkpoint(global_epoch, stage_idx)

                # 主动释放显存碎片，减少OOM和利用率波动
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("\n训练完成!")
        self.save_checkpoint("final", stage_idx if self.cfg.curriculum_enabled else 0)
        self.memory_monitor.print_summary()

    def _git_commit_and_push(self, checkpoint_path: str, epoch):
        """自动 git commit + 异步 push"""
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            # 添加检查点文件和当前代码变更
            subprocess.run(
                ["git", "add", "-f", checkpoint_path],
                cwd=repo_root, check=False, capture_output=True
            )
            subprocess.run(
                ["git", "add", "-A"],
                cwd=repo_root, check=False, capture_output=True
            )
            # commit
            commit_msg = f"checkpoint: epoch {epoch} (stage {self.cfg.board_size}x{self.cfg.board_size})"
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=repo_root, check=False, capture_output=True
            )
            if result.returncode == 0:
                print(f"  [Git] 已提交: {commit_msg}")
                # 异步 push，避免阻塞训练
                subprocess.Popen(
                    ["git", "push", "origin", "HEAD"],
                    cwd=repo_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  [Git] push 已触发 (后台上传)")
            else:
                # 可能是没有变更，忽略
                pass
        except Exception as e:
            print(f"  [Git] 提交失败 (非致命): {e}")

    def save_checkpoint(self, epoch, stage_idx: int = 0):
        path = os.path.join(self.cfg.checkpoint_dir, f"match3_unet_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "stage_idx": stage_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
            "best_iou": self.best_iou,
        }, path)
        print(f"检查点已保存: {path}")
        self._git_commit_and_push(path, epoch)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_iou = checkpoint.get("best_iou", 0.0)
        return checkpoint

    def _visualize_samples(self, samples, epoch: int, stage_size: int):
        """将多个可视化样例垂直拼接成一张大图保存，同时每个样例单独保存CSV。"""
        vis_dir = os.path.join(self.cfg.log_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        panels = []
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        for sample_idx, (board, gt_mask, pred_mask, sample_type) in enumerate(samples):
            board_np = board.argmax(dim=0).cpu().numpy()
            gt_np = gt_mask.squeeze().cpu().numpy()
            pred_prob = torch.sigmoid(pred_mask).squeeze().float().cpu().numpy()
            pred_bin = (pred_prob > self.cfg.mask_threshold).astype(np.int32)

            h, w = board_np.shape
            scale = max(4, 200 // h)
            cell = scale

            def draw_panel(mode):
                img = np.zeros((h * cell, w * cell, 3), dtype=np.uint8)
                for y in range(h):
                    for x in range(w):
                        cy, cx = y * cell, x * cell
                        is_active = False
                        if mode == "input":
                            is_active = True
                        elif mode == "code":
                            is_active = gt_np[y, x] > 0
                        elif mode == "model":
                            is_active = pred_bin[y, x] > 0

                        if is_active:
                            color = FRUIT_COLORS[board_np[y, x] % len(FRUIT_COLORS)]
                        else:
                            color = (0, 0, 0)
                        img[cy:cy+cell, cx:cx+cell] = color

                for y in range(h + 1):
                    img[y*cell:y*cell+1, :] = 80
                for x in range(w + 1):
                    img[:, x*cell:x*cell+1] = 80
                return img

            left = draw_panel("input")
            mid = draw_panel("code")
            right = draw_panel("model")

            gap = np.full((h * cell, 10, 3), 40, dtype=np.uint8)
            board_img = np.concatenate([left, gap, mid, gap, right], axis=1)

            # 标题栏
            header_h = 30
            total_w = board_img.shape[1]
            header = np.full((header_h, total_w, 3), 30, dtype=np.uint8)
            header_pil = Image.fromarray(header)
            draw_h = ImageDraw.Draw(header_pil)

            labels = ["Input Pattern", "Code Label", "Model Pred"]
            panel_w = w * cell
            for i, label in enumerate(labels):
                x_center = i * (panel_w + 10) + panel_w // 2
                bbox = draw_h.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                draw_h.text((x_center - text_w // 2, 5), label, fill=(255, 255, 255), font=font)

            type_text = f"#{sample_idx} {sample_type.upper()}"
            bbox = draw_h.textbbox((0, 0), type_text, font=font_small)
            tw = bbox[2] - bbox[0]
            type_color = (100, 255, 100) if sample_type == "positive" else (255, 100, 100)
            draw_h.text((total_w - tw - 10, header_h - 14), type_text, fill=type_color, font=font_small)

            header = np.array(header_pil)
            combined = np.concatenate([header, board_img], axis=0)
            panels.append(combined)

        # 垂直拼接所有样例，中间加小分隔线
        sep_h = 6
        sep = np.full((sep_h, panels[0].shape[1], 3), 60, dtype=np.uint8)
        final_rows = []
        for i, panel in enumerate(panels):
            final_rows.append(panel)
            if i < len(panels) - 1:
                final_rows.append(sep)
        final_img = np.concatenate(final_rows, axis=0)

        path_png = os.path.join(vis_dir, f"epoch{epoch:03d}_s{stage_size}.png")
        Image.fromarray(final_img).save(path_png)
        print(f"  可视化样例已保存: {path_png} (10个样本垂直拼接)")

        # 合并所有样例的 CSV 数据到一个文件
        merged_csv = [
            f"# Epoch {epoch}, Stage {stage_size}x{stage_size}, TotalSamples=10",
            "# Column mapping: sample_idx=SampleID | sample_type=Pos/Neg | fruit_type=Input | gt_label=Code | pred_prob/pred_label=Model",
            "sample_idx,sample_type,y,x,fruit_type,gt_label,pred_prob,pred_label",
        ]
        for sample_idx, (board, gt_mask, pred_mask, sample_type) in enumerate(samples):
            board_np = board.argmax(dim=0).cpu().numpy()
            gt_np = gt_mask.squeeze().cpu().numpy()
            pred_prob = torch.sigmoid(pred_mask).squeeze().float().cpu().numpy()
            pred_bin = (pred_prob > self.cfg.mask_threshold).astype(np.int32)
            h, w = board_np.shape
            for y in range(h):
                for x in range(w):
                    merged_csv.append(
                        f"{sample_idx},{sample_type},{y},{x},{board_np[y,x]},"
                        f"{int(gt_np[y,x])},{pred_prob[y,x]:.6f},{int(pred_bin[y,x])}"
                    )
        path_csv = os.path.join(vis_dir, f"epoch{epoch:03d}_s{stage_size}.csv")
        with open(path_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(merged_csv))
        print(f"  表格数据已保存: {path_csv} (10个样本合并)")
