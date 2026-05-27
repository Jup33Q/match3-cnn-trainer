"""test_model_new.py - 验证新 CNN-RNN-Transformer 架构"""
import sys
import torch
from config import Match3Config
from models.model import Match3UNet


def test_all():
    print("=" * 60)
    print("CNN-RNN-Transformer 架构验证测试")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: 默认配置前向传播
    print("\n[Test 1] 默认配置前向传播...")
    try:
        cfg = Match3Config()
        model = Match3UNet(cfg)
        x = torch.randn(2, 32, 50, 50)
        y = model(x)
        assert y.shape == (2, 1, 50, 50)
        print(f"  ✅ 输出形状: {y.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 2: 关闭 Transformer
    print("\n[Test 2] 关闭 Transformer...")
    try:
        cfg = Match3Config()
        cfg.use_transformer_output = False
        model = Match3UNet(cfg)
        y = model(x)
        assert y.shape == (2, 1, 50, 50)
        print(f"  ✅ 输出形状: {y.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 3: 关闭 CNN-RNN
    print("\n[Test 3] 关闭 CNN-RNN...")
    try:
        cfg = Match3Config()
        cfg.use_cnn_rnn = False
        model = Match3UNet(cfg)
        y = model(x)
        assert y.shape == (2, 1, 50, 50)
        print(f"  ✅ 输出形状: {y.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 4: 小棋盘 (课程学习 stage1)
    print("\n[Test 4] 小棋盘 10x10...")
    try:
        cfg = Match3Config()
        cfg.board_size = 10
        model = Match3UNet(cfg)
        x4 = torch.randn(2, 32, 10, 10)
        y4 = model(x4)
        assert y4.shape == (2, 1, 10, 10)
        print(f"  ✅ 输出形状: {y4.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 5: 梯度回传
    print("\n[Test 5] 梯度回传...")
    try:
        cfg = Match3Config()
        model = Match3UNet(cfg)
        x5 = torch.randn(2, 32, 50, 50, requires_grad=True)
        y5 = model(x5)
        loss = y5.mean()
        loss.backward()
        assert x5.grad is not None
        print(f"  ✅ 梯度正常")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 6: CUDA + BF16 显存测试 (batch=40, 50x50)
    print("\n[Test 6] CUDA BF16 显存测试 (batch=40, 50x50)...")
    if torch.cuda.is_available():
        try:
            cfg = Match3Config()
            cfg.board_size = 50
            cfg.batch_size = 40
            device = torch.device('cuda')
            model = Match3UNet(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            x6 = torch.randn(40, 32, 50, 50, device=device)
            t6 = torch.randint(0, 2, (40, 1, 50, 50), device=device, dtype=torch.float32)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                p6 = model(x6)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(p6, t6)
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  ✅ 峰值显存: {peak:.2f} GB (限制: 5GB)")
            assert peak < 5.0, f"显存超限: {peak:.2f}GB"
            passed += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed += 1
    else:
        print("  ⚠️ 无 CUDA，跳过")

    # Test 7: 参数量检查
    print("\n[Test 7] 参数量检查...")
    try:
        cfg = Match3Config()
        model = Match3UNet(cfg)
        total = sum(p.numel() for p in model.parameters())
        print(f"  ✅ 总参数量: {total:,} ({total/1e6:.2f}M)")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    ok = test_all()
    sys.exit(0 if ok else 1)
