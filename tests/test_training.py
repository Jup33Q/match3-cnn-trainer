"""test_training.py - 端到端迷你训练测试

验证完整训练流程：
- 数据生成 -> DataLoader
- BF16混合精度训练
- 验证指标计算
- 检查点保存
- TensorBoard日志
"""
import os
import torch
from config import Match3Config
from trainer.trainer import Match3Trainer


def test_mini_training():
    print("=" * 60)
    print("端到端迷你训练测试")
    print("=" * 60)

    config = Match3Config()
    config.board_size = 50
    config.batch_size = 50
    config.precision = "bf16"
    config.train_samples = 100   # 只有2个batch
    config.val_samples = 50      # 1个batch
    config.num_epochs = 1
    config.curriculum_enabled = False
    config.num_workers = 0
    config.save_every_epochs = 1
    config.checkpoint_dir = "./test_checkpoints"
    config.log_dir = "./test_logs"

    # 清理旧测试目录
    import shutil
    for d in [config.checkpoint_dir, config.log_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    print(f"\n配置:")
    print(f"  board_size={config.board_size}")
    print(f"  batch_size={config.batch_size}")
    print(f"  precision={config.precision}")
    print(f"  train_samples={config.train_samples}")
    print(f"  val_samples={config.val_samples}")

    try:
        trainer = Match3Trainer(config)
        trainer.fit()

        # 验证检查点文件存在
        ckpt_files = os.listdir(config.checkpoint_dir)
        assert len(ckpt_files) > 0, "没有生成检查点文件"
        print(f"\n✅ 检查点文件: {ckpt_files}")

        # 验证日志文件存在
        log_files = os.listdir(config.log_dir)
        assert len(log_files) > 0, "没有生成日志文件"
        print(f"✅ 日志文件: {log_files}")

        # 验证推理
        from inference import Match3Predictor
        ckpt_path = os.path.join(config.checkpoint_dir, ckpt_files[0])
        predictor = Match3Predictor(ckpt_path, config)
        import numpy as np
        board = np.random.randint(0, config.num_fruit_types, (50, 50))
        mask = predictor.predict(board)
        assert mask.shape == (50, 50)
        print(f"✅ 推理测试通过, mask shape={mask.shape}, 正样本数={mask.sum()}")

        # 验证评估
        metrics = predictor.evaluate_accuracy(num_test_samples=10)
        print(f"✅ 评估测试通过: {metrics}")

        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ 训练失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理
        for d in [config.checkpoint_dir, config.log_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    import sys
    ok = test_mini_training()
    sys.exit(0 if ok else 1)
