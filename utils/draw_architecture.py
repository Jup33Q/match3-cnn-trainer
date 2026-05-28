"""生成 Match3UNet 架构结构图 (美化版)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

# ==================== 配置 ====================
fig, ax = plt.subplots(1, 1, figsize=(18, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 30)
ax.axis('off')

# 现代配色方案
colors = {
    'input': '#E8F4FD',      # 浅蓝 - 输入
    'stem': '#00B4D8',       # 青蓝 - Stem
    'encoder': '#F94144',    # 红 - Encoder
    'bottleneck': '#F9C74F', # 金黄 - Bottleneck
    'decoder': '#277DA1',    # 深蓝 - Decoder
    'transformer': '#9D4EDD', # 紫 - Transformer
    'output': '#90E0EF',     # 浅青 - Output
    'skip': '#F8961E',       # 橙 - Skip
    'text': '#1D3557',       # 深蓝黑 - 主文字
    'text_light': '#F8F9FA', # 白 - 深色背景上的文字
    'arrow': '#457B9D',      # 箭头
    'bg': '#F1FAEE',         # 薄荷绿背景
    'panel': '#FFFFFF',      # 白色面板
    'border': '#A8DADC',     # 边框
}

fig.patch.set_facecolor(colors['bg'])
ax.set_facecolor(colors['bg'])

# ==================== 标题区域 ====================
ax.text(10, 29.3, 'Match3UNet Architecture', fontsize=32, fontweight='bold',
        ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['border'], linewidth=2))

ax.text(10, 28.5, 'CNN-RNN-Transformer U-Net for Match-3 Pattern Recognition',
        fontsize=14, ha='center', va='center', color='#6C757D', style='italic')

# 统计信息面板
stats_box = FancyBboxPatch((0.8, 27.6), 18.4, 0.7,
                           boxstyle="round,pad=0.05,rounding_size=0.2",
                           facecolor=colors['panel'], edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(stats_box)
ax.text(10, 27.95, 'Total Params: ~176M  |  Input: [B, 4, H, W] (Regular n-gon vertex 3-ch + valid_mask)  |  Output: [B, 1, H, W]',
        fontsize=11, ha='center', va='center', color=colors['text'], fontweight='bold')

# ==================== 辅助函数 ====================
def draw_block(ax, x, y, w, h, color, label, sublabel='', sublabel2='', shadow=True):
    """绘制带阴影的圆角模块框"""
    # 阴影
    if shadow:
        shadow_box = FancyBboxPatch((x - w/2 + 0.08, y - h/2 - 0.08), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor='black', edgecolor='none', alpha=0.08, zorder=1)
        ax.add_patch(shadow_box)
    # 主框
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor=color, edgecolor='white', linewidth=2.5, zorder=2)
    ax.add_patch(box)
    # 标签
    ax.text(x, y + 0.06, label, fontsize=11, ha='center', va='center',
            color=colors['text_light'], fontweight='bold', zorder=3)
    if sublabel:
        ax.text(x, y - 0.22, sublabel, fontsize=8, ha='center', va='center',
                color='white', alpha=0.95, zorder=3)
    if sublabel2:
        ax.text(x, y - 0.42, sublabel2, fontsize=7, ha='center', va='center',
                color='white', alpha=0.85, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color=None, lw=2.0):
    """绘制垂直箭头"""
    color = color or colors['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_skip_arrow(ax, x1, y1, x2, y2, color=None, lw=2.2):
    """绘制跳跃连接（右侧弧线）"""
    color = color or colors['skip']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0.28'))

# ==================== 输入编码区域 ====================
input_box = FancyBboxPatch((0.8, 26.0), 18.4, 0.9,
                           boxstyle="round,pad=0.05,rounding_size=0.15",
                           facecolor=colors['input'], edgecolor='#74C0FC', linewidth=2)
ax.add_patch(input_box)
ax.text(10, 26.45, 'Input Encoding: Regular n-gon vertex 3-ch (cos, sin, 1) + valid_mask  →  [B, 4, H, W]',
        fontsize=11, ha='center', va='center', color='#1971C2', fontweight='bold')

# ==================== 课程学习条 ====================
curr_box = FancyBboxPatch((0.8, 25.0), 18.4, 0.7,
                          boxstyle="round,pad=0.03,rounding_size=0.1",
                          facecolor='#FFF3BF', edgecolor='#FFD43B', linewidth=1.5)
ax.add_patch(curr_box)
ax.text(10, 25.35,
        'Curriculum Learning:  S0(10x10) → S1(25x25) → S2(50x50) → S3(square rand) → S4(rect rand) → S5(rect rand + n=5~12 + match=5~8)',
        fontsize=9, ha='center', va='center', color='#E67700', fontweight='bold')

# ==================== STEM ====================
draw_block(ax, 10, 23.8, 4.0, 1.0, colors['stem'], 'Stem',
           '7x7 Conv + GroupNorm + Swish',
           '[B, 4, H, W] → [B, 32, H, W]')

# ==================== ENCODER ====================
enc_data = [
    ('Encoder 1', '2x ResBlock + MaxPool', '[32, 50, 50] → [64, 25, 25]', 22.3),
    ('Encoder 2', '2x ResBlock + MaxPool', '[64, 25, 25] → [128, 12, 12]', 20.8),
    ('Encoder 3', '2x ResBlock + MaxPool', '[128, 12, 12] → [256, 6, 6]', 19.3),
    ('Encoder 4', '2x ResBlock + MaxPool', '[256, 6, 6] → [512, 3, 3]', 17.8),
    ('Encoder 5', '2x ResBlock + MaxPool', '[512, 3, 3] → [1024, 1, 1]', 16.3),
]

enc_positions = []  # 记录 encoder 位置用于 skip connection
for i, (name, desc, dims, y) in enumerate(enc_data):
    draw_block(ax, 10, y, 4.5, 1.0, colors['encoder'], name, desc, dims)
    enc_positions.append(y)
    if i == 0:
        draw_arrow(ax, 10, 23.8 - 0.5, 10, y + 0.5)
    else:
        draw_arrow(ax, 10, enc_data[i-1][3] - 0.5, 10, y + 0.5)

# ==================== BOTTLENECK ====================
# Pre
draw_block(ax, 10, 14.8, 3.5, 0.7, colors['bottleneck'], 'Bottleneck Pre',
           'ResBlock [1024, 1, 1]')
draw_arrow(ax, 10, 16.3 - 0.5, 10, 14.8 + 0.35)

# Soft-Router 主区域 (带内部结构)
bot_y = 13.3
draw_block(ax, 10, bot_y, 6.0, 1.8, colors['bottleneck'], 'Soft-Router Bottleneck',
           'Router → 3x Parallel CNN Branch → RouterSum',
           '[1024, 1, 1] → [1024, 1, 1]')

# Router 子框 (右侧)
router_w, router_h = 1.4, 0.5
router_x, router_y = 10 + 1.8, bot_y + 0.45
rbox = FancyBboxPatch((router_x - router_w/2, router_y - router_h/2), router_w, router_h,
                       boxstyle="round,pad=0.02", facecolor='#E17055',
                       edgecolor='white', linewidth=2, zorder=3)
ax.add_patch(rbox)
ax.text(router_x, router_y, 'Router', fontsize=8, ha='center', va='center',
        color='white', fontweight='bold', zorder=4)

# 3 个 Branch 子框
branch_labels = ['Branch 1\ndil=1', 'Branch 2\ndil=2', 'Branch 3\ndil=4']
branch_colors = ['#74B9FF', '#A29BFE', '#FD79A8']
for idx, (bl, bc) in enumerate(zip(branch_labels, branch_colors)):
    bx = 10 - 1.8 + idx * 1.8
    by = bot_y - 0.25
    b_w, b_h = 1.4, 0.6
    bbox = FancyBboxPatch((bx - b_w/2, by - b_h/2), b_w, b_h,
                           boxstyle="round,pad=0.02", facecolor=bc,
                           edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(bbox)
    ax.text(bx, by, bl, fontsize=7, ha='center', va='center',
            color='white', fontweight='bold', zorder=4)

draw_arrow(ax, 10, 14.8 - 0.35, 10, bot_y + 0.9)

# Post
draw_block(ax, 10, 11.8, 3.5, 0.7, colors['bottleneck'], 'Bottleneck Post',
           'ResBlock [1024, 1, 1]')
draw_arrow(ax, 10, bot_y - 0.9, 10, 11.8 + 0.35)

# ==================== DECODER ====================
dec_data = [
    ('Decoder 5', 'UpSample + Conv + Concat + 2x ResBlock', '[1024, 1, 1] → [512, 3, 3]', 10.5),
    ('Decoder 4', 'UpSample + Conv + Concat + 2x ResBlock', '[512, 3, 3] → [256, 6, 6]', 9.0),
    ('Decoder 3', 'UpSample + Conv + Concat + 2x ResBlock', '[256, 6, 6] → [128, 12, 12]', 7.5),
    ('Decoder 2', 'UpSample + Conv + Concat + 2x ResBlock', '[128, 12, 12] → [64, 25, 25]', 6.0),
    ('Decoder 1', 'UpSample + Conv + Concat + 2x ResBlock', '[64, 25, 25] → [32, 50, 50]', 4.5),
]

dec_positions = []
for i, (name, desc, dims, y) in enumerate(dec_data):
    draw_block(ax, 10, y, 4.5, 1.0, colors['decoder'], name, desc, dims)
    dec_positions.append(y)
    if i == 0:
        draw_arrow(ax, 10, 11.8 - 0.35, 10, y + 0.5)
    else:
        draw_arrow(ax, 10, dec_data[i-1][3] - 0.5, 10, y + 0.5)

# ==================== TRANSFORMER ====================
draw_block(ax, 10, 3.0, 4.0, 1.0, colors['transformer'], 'Transformer Output Block',
           'Pre-LN Multi-Head Self-Attention + Feed-Forward Network',
           '[B, 32, 50, 50] → [B, 32, 50, 50]')
draw_arrow(ax, 10, 4.5 - 0.5, 10, 3.0 + 0.5)

# ==================== OUTPUT ====================
draw_block(ax, 10, 1.5, 3.5, 0.9, colors['output'], 'Output Head',
           '1x1 Conv (logits) → Sigmoid',
           '[B, 32, 50, 50] → [B, 1, 50, 50]')
draw_arrow(ax, 10, 3.0 - 0.5, 10, 1.5 + 0.45)

# ==================== SKIP CONNECTIONS ====================
skip_pairs = list(zip(enc_positions, reversed(dec_positions)))
for enc_y, dec_y in skip_pairs:
    draw_skip_arrow(ax, 12.25, enc_y, 12.25, dec_y)

# Skip 标签
skip_label_x = 15.8
skip_label_y = 13.3
ax.annotate('Skip Connections\n(Concatenation)', xy=(skip_label_x - 0.3, skip_label_y),
            xytext=(skip_label_x + 1.5, skip_label_y + 1.5),
            fontsize=10, ha='center', va='center', color=colors['skip'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=colors['skip'], lw=1.5,
                           connectionstyle='arc3,rad=-0.2'))

# ==================== ResBlock 详情面板 ====================
detail_y = -0.5
detail_box = FancyBboxPatch((0.8, detail_y), 18.4, 1.8,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            facecolor='white', edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(detail_box)

ax.text(10, detail_y + 1.5, 'ResBlock (BasicBlock) Detail', fontsize=13, ha='center', va='center',
        color=colors['text'], fontweight='bold')

# ResBlock 内部流程图
rb_x = 10
rb_y = detail_y + 0.95
inner_box = FancyBboxPatch((rb_x - 5.5, rb_y - 0.25), 11.0, 0.5,
                            boxstyle="round,pad=0.02", facecolor='#F8F9FA',
                            edgecolor='#DEE2E6', linewidth=1.2)
ax.add_patch(inner_box)
ax.text(rb_x, rb_y,
        'Conv3x3 → GroupNorm → Swish → Conv3x3 → GroupNorm → SE → LayerScale → DropPath → (+Shortcut) → Swish',
        fontsize=9, ha='center', va='center', color=colors['text'])
ax.text(rb_x, rb_y - 0.45,
        'Shortcut: 1x1 Conv + GN (if in_ch ≠ out_ch) else Identity  |  Init: Xavier Normal  |  Normalization: GroupNorm',
        fontsize=8, ha='center', va='center', color='#6C757D')

# ==================== 图例 ====================
legend_items = [
    (colors['input'], 'Input: Regular n-gon vertex 3-ch + valid_mask'),
    (colors['stem'], 'Stem: 7x7 Conv + GN + Swish'),
    (colors['encoder'], 'Encoder: 2x ResBlock + MaxPool'),
    (colors['bottleneck'], 'Bottleneck: Soft-Router + 3x Branch'),
    (colors['decoder'], 'Decoder: UpSample + Concat + 2x ResBlock'),
    (colors['transformer'], 'Transformer: Pre-LN MSA + FFN'),
    (colors['output'], 'Output: 1x1 Conv (logits)'),
]

# 分两行绘制图例
row1 = legend_items[:4]
row2 = legend_items[4:]
for row_idx, row in enumerate([row1, row2]):
    legend_y = detail_y + 0.35 - row_idx * 0.42
    n_items = len(row)
    start_x = 10 - (n_items * 3.2) / 2 + 1.6
    for i, (bg_color, text) in enumerate(row):
        lx = start_x + i * 3.2
        box = FancyBboxPatch((lx - 0.25, legend_y - 0.15), 0.5, 0.3,
                              boxstyle="round,pad=0.02", facecolor=bg_color, edgecolor='white', linewidth=1.5)
        ax.add_patch(box)
        ax.text(lx + 0.4, legend_y, text, fontsize=8, ha='left', va='center', color=colors['text'])

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(__file__), '..')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'architecture_diagram.png'),
            dpi=200, bbox_inches='tight', facecolor=colors['bg'])
print("[PLOT] 架构图已保存至: architecture_diagram.png")
