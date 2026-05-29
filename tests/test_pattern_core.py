"""test_pattern_core.py - 纯NumPy测试PatternDetector核心逻辑

不依赖PyTorch，可在任何Python环境运行，验证:
1. PatternType枚举定义
2. PatternDetector直线检测
3. L/T/Cross型检测
4. 优先级分配
5. 数据生成器逻辑
"""
import sys
import numpy as np
from data.pattern_types import PatternType, PatternDetector


def test_pattern_type_enum():
    """测试PatternType枚举"""
    print("\n[测试] PatternType枚举")
    assert PatternType.NONE.value == 0
    assert PatternType.H3.value == 1
    assert PatternType.H4.value == 2
    assert PatternType.H5.value == 3
    assert PatternType.V3.value == 4
    assert PatternType.V4.value == 5
    assert PatternType.V5.value == 6
    assert PatternType.L.value == 7
    assert PatternType.T.value == 8
    assert PatternType.CROSS.value == 9
    print("  ✓ PatternType枚举正确")


def _make_board(h, w):
    """创建测试棋盘，用交替值避免意外Pattern"""
    board = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            board[y, x] = (y * 7 + x * 13) % 8 + 10  # 确保值>=10，避免与测试值冲突
    return board


def test_horizontal_detection():
    """测试横向直线检测"""
    print("\n[测试] 横向直线检测")
    board = _make_board(5, 10)
    board[2, 1:4] = 1  # H3 at row 2
    board[3, 2:6] = 2  # H4 at row 3
    board[4, 0:5] = 3  # H5 at row 4

    type_map, patterns = PatternDetector.detect_all_patterns(board)

    h3_patterns = [p for p in patterns if p["type"] == PatternType.H3]
    h4_patterns = [p for p in patterns if p["type"] == PatternType.H4]
    h5_patterns = [p for p in patterns if p["type"] == PatternType.H5]

    assert len(h3_patterns) == 1, f"期望1个H3，实际{len(h3_patterns)}: {[p['type'].name for p in patterns]}"
    assert len(h4_patterns) == 1, f"期望1个H4，实际{len(h4_patterns)}: {[p['type'].name for p in patterns]}"
    assert len(h5_patterns) == 1, f"期望1个H5，实际{len(h5_patterns)}: {[p['type'].name for p in patterns]}"

    # H3位置检查
    assert h3_patterns[0]["coords"] == [(2, 1), (2, 2), (2, 3)]
    print("  ✓ 横向直线检测正确")


def test_vertical_detection():
    """测试纵向直线检测"""
    print("\n[测试] 纵向直线检测")
    board = _make_board(10, 5)
    board[1:4, 2] = 1  # V3 at col 2
    board[2:6, 3] = 2  # V4 at col 3
    board[0:5, 4] = 3  # V5 at col 4

    type_map, patterns = PatternDetector.detect_all_patterns(board)

    v3_patterns = [p for p in patterns if p["type"] == PatternType.V3]
    v4_patterns = [p for p in patterns if p["type"] == PatternType.V4]
    v5_patterns = [p for p in patterns if p["type"] == PatternType.V5]

    assert len(v3_patterns) == 1, f"期望1个V3，实际{len(v3_patterns)}: {[p['type'].name for p in patterns]}"
    assert len(v4_patterns) == 1, f"期望1个V4，实际{len(v4_patterns)}: {[p['type'].name for p in patterns]}"
    assert len(v5_patterns) == 1, f"期望1个V5，实际{len(v5_patterns)}: {[p['type'].name for p in patterns]}"
    print("  ✓ 纵向直线检测正确")


def test_l_shape_detection():
    """测试L型检测"""
    print("\n[测试] L型检测")
    board = _make_board(5, 5)
    # L型 (左下拐角): (2,2)为角点，横向向左，纵向向下
    board[2, 0:3] = 1  # H3: (2,0), (2,1), (2,2)
    board[2:5, 2] = 1  # V3: (2,2), (3,2), (4,2)

    type_map, patterns = PatternDetector.detect_all_patterns(board)

    l_patterns = [p for p in patterns if p["type"] == PatternType.L]
    assert len(l_patterns) == 1, f"期望1个L型，实际{len(l_patterns)}"
    assert l_patterns[0]["corner"] == (2, 2)
    assert set(l_patterns[0]["coords"]) == {(2, 0), (2, 1), (2, 2), (3, 2), (4, 2)}
    print("  ✓ L型检测正确")


def test_t_shape_detection():
    """测试T型检测"""
    print("\n[测试] T型检测")
    board = np.zeros((5, 5), dtype=int)
    # T型 (向下): 中心(2,2)，横向3格，纵向向下2格
    board[2, 1:4] = 1  # H3: (2,1), (2,2), (2,3)
    board[2:5, 2] = 1  # V3: (2,2), (3,2), (4,2)
    # 这是一个T型：H3的中点是(2,2)，V3的中点也是(2,2) => 这实际上是十字型！
    # 让我重新设计：T型需要一个端点和一个中点

    # 清除，重新设计 T 向下
    board = np.zeros((5, 5), dtype=int)
    board[2, 1:4] = 1   # H3: (2,1), (2,2), (2,3) 中点=(2,2)
    board[2:5, 2] = 1   # V3: (2,2), (3,2), (4,2) 中点=(3,2)? 不对，V3的ys=[2,3,4]，中点是3
    # 等等，V3的坐标是(2,2),(3,2),(4,2)，sorted ys=[2,3,4]，中点是3。但center是(2,2)
    # 所以 center[0]=2, v3中点=3，不相等。那这是T型吗？
    # 不是，因为center必须同时在两条线上。如果H3中点是(2,2)，那V3必须包含(2,2)。
    # V3包含(2,2),(3,2),(4,2)，那中心是(3,2)吗？不，center=intersection=(2,2)
    # 对于V3来说，(2,2)是端点（因为ys=[2,3,4]，端点是2和4）。
    # 所以 H3中点(2,2) + V3端点(2,2) = T型（向下）

    type_map, patterns = PatternDetector.detect_all_patterns(board)
    t_patterns = [p for p in patterns if p["type"] == PatternType.T]
    cross_patterns = [p for p in patterns if p["type"] == PatternType.CROSS]

    print(f"  检测到的Pattern: {[PatternDetector.PATTERN_NAMES[p['type_id']] for p in patterns]}")

    # 根据优先级，如果同时满足Cross和T，Cross优先级更高
    # 这个结构实际上满足Cross条件吗？
    # H3: (2,1),(2,2),(2,3), xs=[1,2,3], 中点=2 => center[1]=2 == h3_xs[1]=2 ✓
    # V3: (2,2),(3,2),(4,2), ys=[2,3,4], 中点=3 => center[0]=2 != v3_ys[1]=3 ✗
    # 所以不是Cross！
    # 那它是T型吗？
    # is_h_center = center[1]==h3_xs[1] = 2==2 ✓
    # is_v_end = center[0]==v3_ys[0] or center[0]==v3_ys[2] = 2==2 or 2==4 = True ✓
    # => T型！

    assert len(t_patterns) == 1, f"期望1个T型，实际{len(t_patterns)}个: {[PatternDetector.PATTERN_NAMES[p['type_id']] for p in patterns]}"
    print("  ✓ T型检测正确")


def test_cross_detection():
    """测试十字型检测"""
    print("\n[测试] 十字型检测")
    board = _make_board(5, 5)
    # 十字型: 中心(2,2)
    board[2, 1:4] = 1  # H3: (2,1), (2,2), (2,3) 中点=2
    board[1:4, 2] = 1  # V3: (1,2), (2,2), (3,2) 中点=2

    type_map, patterns = PatternDetector.detect_all_patterns(board)

    cross_patterns = [p for p in patterns if p["type"] == PatternType.CROSS]
    print(f"  检测到的Pattern: {[PatternDetector.PATTERN_NAMES[p['type_id']] for p in patterns]}")
    assert len(cross_patterns) == 1, f"期望1个十字型，实际{len(cross_patterns)}"
    assert cross_patterns[0]["center"] == (2, 2)
    print("  ✓ 十字型检测正确")


def test_priority_resolution():
    """测试优先级分配"""
    print("\n[测试] 优先级分配")
    board = _make_board(5, 5)
    # 十字型和T型重叠的场景
    board[2, 1:4] = 1
    board[1:4, 2] = 1

    type_map, patterns = PatternDetector.detect_all_patterns(board)

    # 十字型优先级 > T型
    assert type_map[2, 2] == PatternType.CROSS.value, f"中心(2,2)应为CROSS，实际为{PatternDetector.PATTERN_NAMES.get(type_map[2,2])}"
    print("  ✓ 优先级分配正确")


def test_data_generator():
    """测试数据生成器（纯numpy部分）"""
    print("\n[测试] 数据生成器")
    import sys
    sys.path.insert(0, '.')
    from config import Match3Config
    from data_generator import Match3BoardGenerator

    cfg = Match3Config(board_size=50, num_fruit_types=6)
    gen = Match3BoardGenerator(cfg)

    # 测试easy
    board, mask, pt = gen.generate_board(50, difficulty="easy")
    assert board.shape == (50, 50)
    assert mask.shape == (50, 50)
    assert pt.shape == (50, 50)
    assert board.min() >= 0 and board.max() < cfg.num_fruit_types
    print(f"  easy样本: board={board.shape}, mask正样本比例={mask.mean():.4f}")

    # 测试hard
    board, mask, pt = gen.generate_board(50, difficulty="hard")
    assert mask.sum() == 0, "hard样本不应有消除"
    print(f"  hard样本: 确认无消除Pattern")

    # 测试positive
    board, mask, pt = gen.generate_board(50, difficulty="positive")
    assert mask.sum() > 0, "positive样本应有消除"
    print(f"  positive样本: mask正样本比例={mask.mean():.4f}")

    # 验证一致性: mask=1的位置pattern_type != NONE
    consistent = ((mask == 1) == (pt != PatternType.NONE.value)).all()
    assert consistent, "mask与pattern_type不一致"
    print("  ✓ mask与pattern_type一致性正确")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Match-3 Pattern Core 测试套件")
    print("=" * 60)

    tests = [
        test_pattern_type_enum,
        test_horizontal_detection,
        test_vertical_detection,
        test_l_shape_detection,
        test_t_shape_detection,
        test_cross_detection,
        test_priority_resolution,
        test_data_generator,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 异常: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: 通过={passed}, 失败={failed}, 总计={passed+failed}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
