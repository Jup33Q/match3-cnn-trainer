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
        if args.resume and args.checkpoint:
            ckpt = trainer.load_checkpoint(args.checkpoint)
            start_epoch = ckpt["epoch"] if isinstance(ckpt, dict) else ckpt
            resume_stage = ckpt.get("stage_idx", 0) if isinstance(ckpt, dict) else 0
            print(f"[RESUME] 从 epoch {start_epoch}, stage {resume_stage} 继续训练: {args.checkpoint}")
        # 命令行参数优先级高于检查点
        start_stage = args.start_stage if args.start_stage > 0 else resume_stage
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
