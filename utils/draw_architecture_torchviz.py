"""使用 torchviz 可视化 Match3UNet 计算图

torchviz 基于 graphviz，绘制模型的反向传播计算图（DAG）。
可以直观看到数据流、操作节点和模块层次。

依赖:
    conda install -c conda-forge graphviz
    pip install torchviz
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torchviz import make_dot
from config import Match3Config
from models.model import Match3UNet

# 创建模型和输入
cfg = Match3Config()
model = Match3UNet(cfg)
model.eval()

# 使用 50x50 的 RoPE 输入
dummy_input = torch.randn(1, cfg.fruit_embed_dim, 50, 50)

# 前向传播
output = model(dummy_input)

# ============================================================
# 方式1: 完整计算图 (包含所有梯度节点)
# ============================================================
print("生成完整计算图...")
dot = make_dot(
    output,
    params=dict(model.named_parameters()),
    show_attrs=False,
    show_saved=False
)
# 设置图的方向为从上到下
dot.format = 'png'
dot.attr(rankdir='TB')  # Top to Bottom
dot.attr('node', shape='box', style='rounded,filled', fontsize='9')

dot.render(os.path.join(os.path.dirname(__file__), '..', 'architecture_torchviz_full'), cleanup=False)
print("完整计算图已保存: architecture_torchviz_full.png")

# ============================================================
# 方式2: 简化计算图 (只看模块层次)
# ============================================================
print("\n生成简化计算图 (仅模块层次)...")
dot_simple = make_dot(
    output,
    params=dict(model.named_parameters()),
    show_attrs=False,
    show_saved=False
)
dot_simple.format = 'png'
dot_simple.attr(rankdir='TB')
dot_simple.attr('node', shape='box', style='rounded,filled', fontsize='10')
# 简化: 只显示高层次结构
dot_simple.attr(size='40,60')

dot_simple.render(os.path.join(os.path.dirname(__file__), '..', 'architecture_torchviz_simple'), cleanup=False)
print("简化计算图已保存: architecture_torchviz_simple.png")

# ============================================================
# 方式3: 以模块名称分组的计算图
# ============================================================
print("\n生成按模块分层的计算图...")
# 使用 depth=2 的参数让 graphviz 自动分组
dot_hier = make_dot(
    output,
    params=dict(list(model.named_parameters())[:50]),  # 限制参数数量避免图太大
    show_attrs=False,
    show_saved=False
)
dot_hier.format = 'png'
dot_hier.attr(rankdir='TB')
dot_hier.attr('node', shape='box', style='rounded,filled', fontsize='9')
dot_hier.attr(size='30,40')

dot_hier.render(os.path.join(os.path.dirname(__file__), '..', 'architecture_torchviz_hier'), cleanup=False)
print("分层计算图已保存: architecture_torchviz_hier.png")

print("\n所有 torchviz 可视化完成!")
print("提示: 如果图片太大看不清，可以用: dot -Tsvg architecture_torchviz_full.gv -o full.svg")
