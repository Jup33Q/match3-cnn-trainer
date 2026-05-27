"""生成 Match3UNet 架构结构图 (Soft-Routed Parallel CNN 版)"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(20, 30))
ax.set_xlim(0, 20)
ax.set_ylim(0, 30)
ax.axis('off')

# 颜色定义
colors = {
    'stem': '#4ECDC4',
    'encoder': '#FF6B6B',
    'bottleneck': '#FFE66D',
    'decoder': '#1A535C',
    'output': '#95E1D3',
    'skip': '#F7FFF7',
    'text': '#2D3436',
    'arrow': '#636e72',
    'bg': '#f8f9fa',
    'rope': '#9b59b6',
    'stage': '#3498db',
    'router': '#e17055',
    'branch': '#fdcb6e',
}

fig.patch.set_facecolor(colors['bg'])
ax.set_facecolor(colors['bg'])

# 标题
ax.text(10, 29.5, 'Match3UNet Architecture', fontsize=28, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(10, 28.8, 'Soft-Routed Parallel CNN U-Net for Match-3 Pattern Recognition (RoPE Input, 5-Stage Curriculum)',
        fontsize=14, ha='center', va='center', color='#636e72')

# 统计信息框
stats_box = FancyBboxPatch((0.5, 27.5), 19, 1.0, boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor='white', edgecolor='#dfe6e9', linewidth=1.5)
ax.add_patch(stats_box)
ax.text(10, 28.0, 'Total Params: ~176M (Soft-Router + Transformer)  |  Input: [B, 32, H, W] (Fruit RoPE)  |  Output: [B, 1, H, W]',
        fontsize=12, ha='center', va='center', color=colors['text'], fontweight='bold')

# ============ Fruit RoPE 编码框 ============
rope_box = FancyBboxPatch((0.5, 26.0), 19, 0.9, boxstyle="round,pad=0.05,rounding_size=0.15",
                           facecolor=colors['rope'], edgecolor='white', linewidth=2, alpha=0.15)
ax.add_patch(rope_box)
ax.text(10, 26.45, 'Fruit RoPE Encoding: fruit_type_id → 32-dim vector  (supports 5~12 dynamic fruit types)',
        fontsize=11, ha='center', va='center', color=colors['rope'], fontweight='bold')

def draw_block(ax, x, y, w, h, color, label, sublabel='', alpha=1.0):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='white', linewidth=2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y + 0.05, label, fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')
    if sublabel:
        ax.text(x, y - 0.25, sublabel, fontsize=7, ha='center', va='center',
                color='white', alpha=0.9)

def draw_arrow(ax, x1, y1, x2, y2, color='#636e72', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_skip(ax, x1, y1, x2, y2, color='#fdcb6e'):
    # 绘制跳跃连接弧线
    mid_x = (x1 + x2) / 2 + 2.5
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                               connectionstyle=f'arc3,rad=0.35'))

# ============ STEM ============
draw_block(ax, 10, 24.5, 3.5, 0.9, colors['stem'], 'Stem',
           'Conv7×7 → GN → Swish\n[32, H, W] → [32, H, W]')

# ============ ENCODER ============
enc_stages = [
    ('Encoder 1', '[32,50,50]→[64,25,25]', '2×ResBlock + MaxPool', 23.0),
    ('Encoder 2', '[64,25,25]→[128,12,12]', '2×ResBlock + MaxPool', 21.5),
    ('Encoder 3', '[128,12,12]→[256,6,6]', '2×ResBlock + MaxPool', 20.0),
    ('Encoder 4', '[256,6,6]→[512,3,3]', '2×ResBlock + MaxPool', 18.5),
    ('Encoder 5', '[512,3,3]→[1024,1,1]', '2×ResBlock + MaxPool', 17.0),
]

for i, (name, dims, desc, y) in enumerate(enc_stages):
    draw_block(ax, 10, y, 4.2, 0.9, colors['encoder'], name, f'{desc}\n{dims}')
    draw_arrow(ax, 10, y + 0.9 + 0.15, 10, y + 0.45)

# ============ BOTTLENECK (Soft-Router) ============
# bottleneck_pre
pre_y = 15.8
draw_block(ax, 10, pre_y, 3.0, 0.6, colors['bottleneck'], 'bottleneck_pre',
           'ResBlock [1024,1,1]')
draw_arrow(ax, 10, 17.0 - 0.45, 10, pre_y + 0.3)

# Main bottleneck area
bot_y = 14.3
draw_block(ax, 10, bot_y, 5.5, 1.6, colors['bottleneck'], 'Soft-Router Bottleneck',
           'Router + 3×Parallel CNN Branch + RouterSum\n[1024,1,1] → [1024,1,1]')

# Router sub-box (right side)
router_box = FancyBboxPatch((10 + 1.4, bot_y + 0.18), 1.3, 0.42,
                             boxstyle="round,pad=0.02", facecolor=colors['router'],
                             edgecolor='white', linewidth=1.5, alpha=0.95)
ax.add_patch(router_box)
ax.text(10 + 2.05, bot_y + 0.39, 'Router', fontsize=7, ha='center', va='center',
        color='white', fontweight='bold')

# Branch sub-boxes (left side inside bottleneck)
branch_labels = ['Branch1\ndil=1', 'Branch2\ndil=2', 'Branch3\ndil=4']
for idx, bl in enumerate(branch_labels):
    bx = 10 - 1.6 + idx * 1.15
    by = bot_y - 0.22
    b_box = FancyBboxPatch((bx - 0.48, by - 0.22), 0.96, 0.44,
                            boxstyle="round,pad=0.02", facecolor=colors['branch'],
                            edgecolor='white', linewidth=1.2, alpha=0.9)
    ax.add_patch(b_box)
    ax.text(bx, by, bl, fontsize=7, ha='center', va='center',
            color='white', fontweight='bold')

draw_arrow(ax, 10, pre_y - 0.3, 10, bot_y + 0.8)

# bottleneck_post
post_y = 13.0
draw_block(ax, 10, post_y, 3.0, 0.6, colors['bottleneck'], 'bottleneck_post',
           'ResBlock [1024,1,1]')
draw_arrow(ax, 10, bot_y - 0.8, 10, post_y + 0.3)

# ============ DECODER ============
dec_stages = [
    ('Decoder 5', '[1024,1,1]→[512,3,3]', 'UpSample + Conv + Concat + 2×ResBlock', 11.7),
    ('Decoder 4', '[512,3,3]→[256,6,6]', 'UpSample + Conv + Concat + 2×ResBlock', 10.2),
    ('Decoder 3', '[256,6,6]→[128,12,12]', 'UpSample + Conv + Concat + 2×ResBlock', 8.7),
    ('Decoder 2', '[128,12,12]→[64,25,25]', 'UpSample + Conv + Concat + 2×ResBlock', 7.2),
    ('Decoder 1', '[64,25,25]→[32,50,50]', 'UpSample + Conv + Concat + 2×ResBlock', 5.7),
]

for i, (name, dims, desc, y) in enumerate(dec_stages):
    draw_block(ax, 10, y, 4.2, 0.9, colors['decoder'], name, f'{desc}\n{dims}')
    if i == 0:
        draw_arrow(ax, 10, post_y - 0.3, 10, y + 0.45)
    else:
        draw_arrow(ax, 10, y + 0.9 + 0.15 + 0.6, 10, y + 0.45)

# ============ TRANSFORMER OUTPUT BLOCK ============
draw_block(ax, 10, 4.0, 3.5, 0.9, colors['stage'], 'Transformer Block',
           'MSA + FFN\n[32, 50, 50] → [32, 50, 50]')
draw_arrow(ax, 10, 4.25, 10, 4.45)

# ============ OUTPUT HEAD ============
draw_block(ax, 10, 2.5, 3.5, 0.9, colors['output'], 'Output Head',
           'Conv 1×1 (logits)\n[32, 50, 50] → [1, 50, 50]')
draw_arrow(ax, 10, 3.55, 10, 2.95)

# ============ SKIP CONNECTIONS ============
skip_pairs = [
    (23.0, 5.7),   # Enc1 -> Dec1
    (21.5, 7.2),   # Enc2 -> Dec2
    (20.0, 8.7),   # Enc3 -> Dec3
    (18.5, 10.2),  # Enc4 -> Dec4
    (17.0, 11.7),  # Enc5 -> Dec5
]

for enc_y, dec_y in skip_pairs:
    draw_skip(ax, 12.1, enc_y, 12.1, dec_y)

# Skip connection 标签
ax.text(15.5, 14.35, 'Skip Connections\n(Concatenation)', fontsize=10, ha='center', va='center',
        color='#d63031', fontweight='bold', style='italic')

# ============ LEGEND ============
legend_items = [
    (colors['stem'], 'Stem: 7×7 Conv + GN + Swish'),
    (colors['encoder'], 'Encoder: 2×ResBlock + MaxPool2d'),
    (colors['bottleneck'], 'Bottleneck: Soft-Router + 3×Parallel CNN'),
    (colors['decoder'], 'Decoder: UpSample + Concat + 2×ResBlock'),
    (colors['stage'], 'Transformer: Pre-LN MSA + FFN'),
    (colors['output'], 'Output: 1×1 Conv (logits)'),
]

legend_y = 1.5
for i, (color, text) in enumerate(legend_items):
    lx = 1.8 + i * 3.1
    box = FancyBboxPatch((lx - 0.25, legend_y - 0.15), 0.5, 0.3,
                          boxstyle="round,pad=0.02", facecolor=color, edgecolor='white')
    ax.add_patch(box)
    ax.text(lx + 0.4, legend_y, text, fontsize=8, ha='left', va='center', color=colors['text'])

# ============ RESBLOCK DETAIL ============
detail_box = FancyBboxPatch((0.5, 0.0), 19, 1.2, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor='white', edgecolor='#dfe6e9', linewidth=1.5)
ax.add_patch(detail_box)
ax.text(10, 0.95, 'ResBlock (BasicBlock) Detail', fontsize=12, ha='center', va='center',
        color=colors['text'], fontweight='bold')

# 画 ResBlock 内部结构示意
rb_x = 5.0
rb_y = 0.45
ax.add_patch(FancyBboxPatch((rb_x - 2.2, rb_y - 0.3), 4.4, 0.6,
                             boxstyle="round,pad=0.02", facecolor='#ecf0f1', edgecolor='#b2bec3'))
ax.text(rb_x, rb_y, 'Conv3×3 → GN → Swish → Conv3×3 → GN → SE → LayerScale → DropPath → (+Shortcut) → Swish',
        fontsize=9, ha='center', va='center', color=colors['text'])
ax.text(rb_x, rb_y - 0.5, 'Shortcut: 1×1 Conv + GN (if in_ch ≠ out_ch) else Identity  |  Init: Xavier Normal',
        fontsize=8, ha='center', va='center', color='#636e72')

# ============ 课程学习5阶段说明 ============
stage_box = FancyBboxPatch((0.5, 25.0), 19, 0.7, boxstyle="round,pad=0.03,rounding_size=0.1",
                            facecolor=colors['stage'], edgecolor='white', linewidth=1.5, alpha=0.12)
ax.add_patch(stage_box)
stage_text = (
    'Curriculum:  S1(10×10, 6fruit, match3~5) → S2(25×25) → S3(50×50) → '
    'S4(rand 10~50) → S5(rand 10~50, rand 5~12 fruit, match5~8)'
)
ax.text(10, 25.35, stage_text, fontsize=9, ha='center', va='center',
        color=colors['stage'], fontweight='bold')

plt.tight_layout()
import os
save_dir = os.path.join(os.path.dirname(__file__), '..')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'architecture_diagram.png'),
            dpi=200, bbox_inches='tight', facecolor=colors['bg'])
print("架构图已保存至: architecture_diagram.png")
