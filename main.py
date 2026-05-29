"""main.py - CLI入口"""
import argparse
import json
import numpy as np

from config import Match3Config
from trainer.trainer import Match3Trainer
from trainer.inference import Match3Predictor


def main():
    parser = argparse.ArgumentParser(description="Match-3 Pattern Recognition Trainer (50x50)")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--config", type=str, help="JSON config file path")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint for eval/infer/resume")
    parser.add_argument("--board", type=str, help="Numpy board file for inference")
    parser.add_argument("--resume", action="store_true", help="从 --checkpoint 继续训练")
    parser.add_argument("--start_stage", type=int, default=0, help="课程学习从第几个阶段开始 (0-based)")
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
        resume_stage = 0
        start_epoch = 0
        epochs_per_stage = config.curriculum_epochs_per_stage if config.curriculum_enabled else config.num_epochs

        if args.resume and args.checkpoint:
            ckpt = trainer.load_checkpoint(args.checkpoint)
            raw_epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else ckpt
            resume_stage = ckpt.get("stage_idx", 0) if isinstance(ckpt, dict) else 0

            # 处理 best/final 等字符串 epoch，转为该 stage 的起始 epoch
            if isinstance(raw_epoch, str) or not isinstance(raw_epoch, int):
                start_epoch = resume_stage * epochs_per_stage
                print(f"[RESUME] 检查点 epoch='{raw_epoch}'，转为 stage {resume_stage} 起始 epoch={start_epoch}")
            else:
                start_epoch = raw_epoch
            print(f"[RESUME] 检查点记录: epoch={start_epoch}, stage={resume_stage}: {args.checkpoint}")

        # 命令行参数优先级高于检查点
        if args.start_stage > 0:
            start_stage = args.start_stage
            # 显式跳 stage 时，epoch 也同步到该 stage 起点
            start_epoch = start_stage * epochs_per_stage
            print(f"[STAGE] 显式指定从 stage {start_stage} 开始，epoch 设为 {start_epoch}")
        elif args.resume and args.checkpoint:
            # 自动推断：若当前stage已完成（epoch为该stage最后一个），则进入下一个stage
            if isinstance(start_epoch, int) and start_epoch > 0 and start_epoch % epochs_per_stage == 0:
                start_stage = resume_stage + 1
                print(f"[RESUME] 检测到 stage {resume_stage} 已完成，自动进入 stage {start_stage}")
            else:
                start_stage = resume_stage
        else:
            start_stage = resume_stage

        # 边界检查
        total_stages = len(config.curriculum_stages) if config.curriculum_enabled else 1
        if start_stage >= total_stages:
            print(f"[WARN] start_stage={start_stage} 超出课程学习阶段总数 {total_stages}，将调整为 {total_stages - 1}")
            start_stage = total_stages - 1

        trainer.fit(start_stage=start_stage, start_epoch=start_epoch)

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
