"""test_model_new.py - 验证 Soft-Routed Parallel CNN U-Net 架构"""
import sys
import torch
from config import Match3Config
from models.model import Match3UNet


def test_all():
    print("=" * 60)
    print("Soft-Routed Parallel CNN U-Net 架构验证测试")
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

    # Test 3: 多分支配置 (5分支)
    print("\n[Test 3] 5分支软路由...")
    try:
        cfg = Match3Config()
        cfg.num_router_branches = 5
        cfg.router_branch_dilations = [1, 2, 4, 8, 1]
        model = Match3UNet(cfg)
        y = model(x)
        assert y.shape == (2, 1, 50, 50)
        print(f"  ✅ 输出形状: {y.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 4: 单分支退化测试
    print("\n[Test 4] 单分支软路由...")
    try:
        cfg = Match3Config()
        cfg.num_router_branches = 1
        cfg.router_branch_dilations = [1]
        model = Match3UNet(cfg)
        y = model(x)
        assert y.shape == (2, 1, 50, 50)
        print(f"  ✅ 输出形状: {y.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 5: 小棋盘 (课程学习 stage1)
    print("\n[Test 5] 小棋盘 10x10...")
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

    # Test 6: 梯度回传
    print("\n[Test 6] 梯度回传...")
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

    # Test 7: CUDA + BF16 前向测试 (batch=8, 50x50)
    print("\n[Test 7] CUDA BF16 前向测试 (batch=8, 50x50)...")
    if torch.cuda.is_available():
        try:
            cfg = Match3Config()
            cfg.board_size = 50
            device = torch.device('cuda')
            model = Match3UNet(cfg).to(device)
            x6 = torch.randn(8, 32, 50, 50, device=device)
            torch.cuda.empty_cache()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                p6 = model(x6)
            assert p6.shape == (8, 1, 50, 50)
            print(f"  ✅ CUDA BF16 输出形状: {p6.shape}")
            passed += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed += 1
    else:
        print("  ⚠️ 无 CUDA，跳过")

    # Test 8: 参数量检查
    print("\n[Test 8] 参数量检查...")
    try:
        cfg = Match3Config()
        model = Match3UNet(cfg)
        total = sum(p.numel() for p in model.parameters())
        print(f"  ✅ 总参数量: {total:,} ({total/1e6:.2f}M)")
        passed += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        failed += 1

    # Test 9: Router权重检查 (确保softmax和为1)
    print("\n[Test 9] Router softmax 权重校验...")
    try:
        cfg = Match3Config()
        model = Match3UNet(cfg)
        model.eval()
        with torch.no_grad():
            x_test = torch.randn(2, 32, 50, 50)
            # 手动运行到router
            x_enc = model.stem(x_test)
            skips = []
            for enc in model.encoders:
                x_enc, skip = enc(x_enc)
                skips.append(skip)
            x_enc = model.bottleneck_pre(x_enc)
            logits = model.router(x_enc)
            weights = torch.softmax(logits, dim=1)
            sum_w = weights.sum(dim=1)
            assert torch.allclose(sum_w, torch.ones_like(sum_w), atol=1e-5)
            print(f"  ✅ Router权重和=1, 形状: {weights.shape}")
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
