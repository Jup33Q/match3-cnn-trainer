"""resume_train.py - 从 Stage 检查点恢复训练

用法:
    # 指定检查点恢复（自动推断下一个stage）
    python scripts/resume_train.py --checkpoint ./checkpoints/match3_unet_epochstage2_epoch60.pt

    # 从 best 检查点恢复，手动指定从 stage 3 开始
    python scripts/resume_train.py --checkpoint ./checkpoints/match3_unet_epochbest.pt --start_stage 3

    # 自动寻找最新的 stage 检查点
    python scripts/resume_train.py --auto --checkpoint_dir ./checkpoints

    # 覆盖配置参数
    python scripts/resume_train.py --checkpoint stage2.pt --config my_config.json --lr 5e-4
"""
import argparse
import glob
import json
import os
import re
import sys

# 把项目根目录加入 PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Match3Config
from trainer.trainer import Match3Trainer


def parse_stage_epoch_from_filename(filename: str):
    """从文件名解析 stage_idx 和 global_epoch

    支持格式:
        match3_unet_epochstage2_epoch60.pt  -> (2, 60)
        match3_unet_epoch45.pt              -> (None, 45)
        match3_unet_epochbest.pt            -> (None, None)
    """
    base = os.path.basename(filename)
    # 匹配 stageX_epochY
    m = re.search(r"stage(\d+)_epoch(\d+)", base)
    if m:
        return int(m.group(1)), int(m.group(2))
    # 匹配纯 epoch 数字
    m = re.search(r"epoch(\d+)\.pt", base)
    if m:
        return None, int(m.group(1))
    return None, None


def find_latest_stage_checkpoint(checkpoint_dir: str):
    """自动寻找 checkpoint 目录下最新的 stage 检查点"""
    pattern = os.path.join(checkpoint_dir, "match3_unet_epochstage*_epoch*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    # 按修改时间排序，取最新的
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main():
    parser = argparse.ArgumentParser(description="从 Stage 检查点恢复 Match-3 训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="要恢复的检查点文件路径")
    parser.add_argument("--auto", action="store_true",
                        help="自动寻找 checkpoint_dir 下最新的 stage 检查点")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="检查点目录（--auto 时使用）")
    parser.add_argument("--start_stage", type=int, default=None,
                        help="手动指定从第几个 stage 开始 (0-based)。"
                             "默认 = 检查点中的 stage_idx + 1（即下一个stage）")
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="手动指定 global epoch。默认 = 检查点中的 epoch")
    parser.add_argument("--config", type=str, help="JSON 配置文件路径")
    parser.add_argument("--lr", type=float, help="覆盖学习率")
    args = parser.parse_args()

    # ----- 确定检查点路径 -----
    ckpt_path = args.checkpoint
    if args.auto:
        ckpt_path = find_latest_stage_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            print(f"[ERROR] 在 {args.checkpoint_dir} 中未找到 stage 检查点")
            sys.exit(1)
        print(f"[AUTO] 找到最新 stage 检查点: {ckpt_path}")

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[ERROR] 检查点不存在: {ckpt_path}")
        print("提示: 使用 --auto 自动寻找，或手动指定 --checkpoint")
        sys.exit(1)

    # ----- 加载配置 -----
    config = Match3Config()
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    # 命令行覆盖学习率
    if args.lr is not None:
        config.learning_rate = args.lr
        print(f"[OVERRIDE] 学习率设为 {args.lr}")

    # ----- 初始化 Trainer 并加载检查点 -----
    trainer = Match3Trainer(config)
    ckpt = trainer.load_checkpoint(ckpt_path)

    # ----- 解析恢复位置 -----
    file_stage, file_epoch = parse_stage_epoch_from_filename(ckpt_path)

    # 优先使用检查点内部保存的 stage_idx，其次从文件名解析
    ckpt_stage = ckpt.get("stage_idx", file_stage)
    ckpt_epoch = ckpt.get("epoch", file_epoch)

    # 处理 epoch 可能是字符串（如 "stage2_epoch60" 或 "best"）
    if isinstance(ckpt_epoch, str):
        # 尝试从中提取数字
        m = re.search(r"epoch(\d+)", ckpt_epoch)
        if m:
            ckpt_epoch = int(m.group(1))
        else:
            ckpt_epoch = file_epoch if file_epoch is not None else 0

    # 计算默认恢复位置：从**下一个** stage 开始
    default_start_stage = (ckpt_stage + 1) if ckpt_stage is not None else 0
    default_start_epoch = ckpt_epoch if ckpt_epoch is not None else 0

    start_stage = args.start_stage if args.start_stage is not None else default_start_stage
    start_epoch = args.start_epoch if args.start_epoch is not None else default_start_epoch

    print(f"\n{'='*60}")
    print(f"[RESUME] 检查点: {ckpt_path}")
    print(f"[RESUME] 检查点记录: stage={ckpt_stage}, epoch={ckpt_epoch}, best_iou={ckpt.get('best_iou', 0):.4f}")
    print(f"[RESUME] 将从 stage={start_stage}, global_epoch={start_epoch} 继续训练")
    print(f"{'='*60}\n")

    # ----- 开始训练 -----
    trainer.fit(start_stage=start_stage, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
