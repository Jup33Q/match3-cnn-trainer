"""生成 Match3UNet 默认架构结构图"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(20, 26))
ax.set_xlim(0, 20)
ax.set_ylim(0, 26)
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
    'bg': '#f8f9fa'
}

fig.patch.set_facecolor(colors['bg'])
ax.set_facecolor(colors['bg'])

# 标题
ax.text(10, 25.5, 'Match3UNet Architecture', fontsize=28, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(10, 24.8, 'Deep ResNet-U-Net for Match-3 Pattern Recognition (50×50 Input)',
        fontsize=14, ha='center', va='center', color='#636e72')

# 统计信息框
stats_box = FancyBboxPatch((0.5, 23.2), 19, 1.2, boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor='white', edgecolor='#dfe6e9', linewidth=1.5)
ax.add_patch(stats_box)
ax.text(10, 23.8, 'Total Params: 174.4M  |  Model Size (FP32): 697.5 MB  |  Input: [B, 6, 50, 50]  |  Output: [B, 1, 50, 50]',
        fontsize=12, ha='center', va='center', color=colors['text'], fontweight='bold')

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
draw_block(ax, 10, 22.5, 3.5, 0.9, colors['stem'], 'Stem',
           'Conv7×7 → BN → ReLU\n[6, 50, 50] → [32, 50, 50]')

# ============ ENCODER ============
enc_stages = [
    ('Encoder 1', '[32,50,50]→[64,25,25]', '3×ResBlock + MaxPool', 21.0),
    ('Encoder 2', '[64,25,25]→[128,12,12]', '3×ResBlock + MaxPool', 19.5),
    ('Encoder 3', '[128,12,12]→[256,6,6]', '3×ResBlock + MaxPool', 18.0),
    ('Encoder 4', '[256,6,6]→[512,3,3]', '3×ResBlock + MaxPool', 16.5),
    ('Encoder 5', '[512,3,3]→[1024,1,1]', '3×ResBlock + MaxPool', 15.0),
]

for i, (name, dims, desc, y) in enumerate(enc_stages):
    draw_block(ax, 10, y, 4.2, 0.9, colors['encoder'], name, f'{desc}\n{dims}')
    draw_arrow(ax, 10, y + 0.9 + 0.15, 10, y + 0.45)

# ============ BOTTLENECK ============
draw_block(ax, 10, 13.2, 4.0, 1.0, colors['bottleneck'], 'Bottleneck',
           '4× BasicBlock (ResNet)\n[1024, 1, 1] → [1024, 1, 1]')
draw_arrow(ax, 10, 14.55, 10, 13.7)

# ============ DECODER ============
dec_stages = [
    ('Decoder 5', '[1024,1,1]→[512,3,3]', 'UpConv + Concat + 3×ResBlock', 11.7),
    ('Decoder 4', '[512,3,3]→[256,6,6]', 'UpConv + Concat + 3×ResBlock', 10.2),
    ('Decoder 3', '[256,6,6]→[128,12,12]', 'UpConv + Concat + 3×ResBlock', 8.7),
    ('Decoder 2', '[128,12,12]→[64,25,25]', 'UpConv + Concat + 3×ResBlock', 7.2),
    ('Decoder 1', '[64,25,25]→[32,50,50]', 'UpConv + Concat + 3×ResBlock', 5.7),
]

for i, (name, dims, desc, y) in enumerate(dec_stages):
    draw_block(ax, 10, y, 4.2, 0.9, colors['decoder'], name, f'{desc}\n{dims}')
    if i == 0:
        draw_arrow(ax, 10, 12.65, 10, y + 0.45)
    else:
        draw_arrow(ax, 10, y + 0.9 + 0.15 + 0.6, 10, y + 0.45)

# ============ OUTPUT HEAD ============
draw_block(ax, 10, 4.2, 3.5, 0.9, colors['output'], 'Output Head',
           'Conv 1×1 (no sigmoid)\n[32, 50, 50] → [1, 50, 50]')
draw_arrow(ax, 10, 5.25, 10, 4.65)

# ============ SKIP CONNECTIONS ============
# 从每个 Encoder 到对应的 Decoder
skip_pairs = [
    (21.0, 5.7),   # Enc1 -> Dec1
    (19.5, 7.2),   # Enc2 -> Dec2
    (18.0, 8.7),   # Enc3 -> Dec3
    (16.5, 10.2),  # Enc4 -> Dec4
    (15.0, 11.7),  # Enc5 -> Dec5
]

for enc_y, dec_y in skip_pairs:
    draw_skip(ax, 12.1, enc_y, 12.1, dec_y)

# Skip connection 标签
ax.text(15.5, 13.35, 'Skip Connections\n(Concatenation)', fontsize=10, ha='center', va='center',
        color='#d63031', fontweight='bold', style='italic')

# ============ LEGEND ============
legend_items = [
    (colors['stem'], 'Stem: 7×7 Conv + BN + ReLU'),
    (colors['encoder'], 'Encoder: 3×ResBlock + MaxPool2d'),
    (colors['bottleneck'], 'Bottleneck: 4×ResBlock'),
    (colors['decoder'], 'Decoder: UpConv + Concat + 3×ResBlock'),
    (colors['output'], 'Output: 1×1 Conv (logits)'),
]

legend_y = 2.8
for i, (color, text) in enumerate(legend_items):
    lx = 2.5 + i * 3.5
    box = FancyBboxPatch((lx - 0.25, legend_y - 0.15), 0.5, 0.3,
                          boxstyle="round,pad=0.02", facecolor=color, edgecolor='white')
    ax.add_patch(box)
    ax.text(lx + 0.4, legend_y, text, fontsize=8, ha='left', va='center', color=colors['text'])

# ============ RESBLOCK DETAIL ============
detail_box = FancyBboxPatch((0.5, 0.3), 19, 1.8, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor='white', edgecolor='#dfe6e9', linewidth=1.5)
ax.add_patch(detail_box)
ax.text(10, 1.85, 'ResBlock (BasicBlock) Detail', fontsize=12, ha='center', va='center',
        color=colors['text'], fontweight='bold')

# 画 ResBlock 内部结构示意
rb_x = 4.5
rb_y = 1.15
ax.add_patch(FancyBboxPatch((rb_x - 1.8, rb_y - 0.35), 3.6, 0.7,
                             boxstyle="round,pad=0.02", facecolor='#ecf0f1', edgecolor='#b2bec3'))
ax.text(rb_x, rb_y, 'Conv3×3 → BN → ReLU → Conv3×3 → BN → (+Shortcut) → ReLU',
        fontsize=9, ha='center', va='center', color=colors['text'])
ax.text(rb_x, rb_y - 0.55, 'Shortcut: 1×1 Conv + BN (if in_ch ≠ out_ch) else Identity',
        fontsize=8, ha='center', va='center', color='#636e72')

# 右侧说明
ax.text(14, 1.4, '• Each Encoder/Decoder stage contains 3 ResBlocks\n'
                  '• Bottleneck contains 4 ResBlocks\n'
                  '• Optional Mamba2DLayer can replace odd-index bottleneck blocks',
        fontsize=9, ha='left', va='center', color=colors['text'])

plt.tight_layout()
plt.savefig('/home/jup33q/vllm_scripts/super_mirror/match3_cnn_trainer/architecture_diagram.png',
            dpi=200, bbox_inches='tight', facecolor=colors['bg'])
print("架构图已保存至: architecture_diagram.png")
