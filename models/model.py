"""model.py - 深度 ResNet-U-Net (5层, 每stage 3×ResBlock, 4层Bottleneck)

架构升级:
- 5层 Encoder/Decoder (50x50 -> 25x25 -> 12x12 -> 6x6 -> 3x3 -> 1x1)
- 每个 Encoder/Decoder stage 包含 **3个 BasicBlock** (ResNet风格)
- Bottleneck 包含 **4个 BasicBlock**
- 总参数量 ~174M, BF16训练显存 ~3.27GB
- 激活函数使用 SiLU (Swish), 避免 inplace ReLU 导致的 segfault
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Match3Config
from .mamba_layer import Mamba2DLayer


class BasicBlock(nn.Module):
    """ResNet BasicBlock: 2x(3x3) + 残差连接 + 可选 Dropout (SiLU激活)"""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = (3 // 2) * dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=padding,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=padding,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class EncoderBlock(nn.Module):
    """Encoder Stage: N× BasicBlock + MaxPool + Skip Connection"""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 3,
                 use_dilation: bool = False, dilation_rate: int = 1, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            ich = in_ch if i == 0 else out_ch
            dil = dilation_rate if use_dilation else 1
            self.blocks.append(BasicBlock(ich, out_ch, dil, dropout))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # 课程学习阶段小尺寸输入(如10x10)经过多层池化后可能变为1x1，
        # 继续MaxPool2d(2,2)会产生0x0输出导致崩溃。此时跳过池化。
        if min(x.shape[2:]) <= 1:
            return x, x
        return self.pool(x), x  # pooled, skip


class DecoderBlock(nn.Module):
    """Decoder Stage: Upsample + Concat + N× BasicBlock"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            ich = out_ch + skip_ch if i == 0 else out_ch
            self.blocks.append(BasicBlock(ich, out_ch, dropout=dropout))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x)
        return x


class Match3UNet(nn.Module):
    """深度三消 Pattern 识别 ResNet-U-Net + 可选 Mamba 层

    架构:
    - 5层 Encoder (每stage 3× BasicBlock)
    - 4层 Bottleneck (2× BasicBlock + 2× Mamba2DLayer, 可选)
    - 5层 Decoder (每stage 3× BasicBlock)
    - 首层 7x7 大卷积核
    - 输出 logits (无sigmoid, 由loss处理)

    规模 (base=32, blocks=5, stage_blocks=3, bot_blocks=4):
    - 基础参数量: ~174M
    - 启用Mamba后: ~241M
    - BF16训练显存 (batch=200): ~7-10GB
    """

    def __init__(self, config: Match3Config):
        super().__init__()
        self.cfg = config
        n_stage = config.get("blocks_per_stage", 2)
        n_bot = config.get("bottleneck_blocks", 2)

        # Stem (支持 RoPE 编码输入)
        input_ch = getattr(config, 'fruit_embed_dim', config.num_fruit_types)
        self.stem = nn.Sequential(
            nn.Conv2d(input_ch, config.base_channels,
                      config.initial_kernel_size,
                      padding=config.initial_kernel_size // 2, bias=False),
            nn.BatchNorm2d(config.base_channels),
            nn.SiLU()
        )

        dropout = getattr(config, 'dropout', 0.0)

        # Encoder (5 stages)
        self.encoders = nn.ModuleList()
        ch = config.base_channels
        for i in range(config.num_encoder_blocks):
            next_ch = ch * 2
            dilation = config.dilation_rates[i] if config.use_dilation and i < len(config.dilation_rates) else 1
            self.encoders.append(EncoderBlock(ch, next_ch, n_stage, config.use_dilation, dilation, dropout))
            ch = next_ch

        # Bottleneck (BasicBlock + Mamba2DLayer 混合)
        if config.use_mamba:
            self.bottleneck = nn.ModuleList()
            for i in range(n_bot):
                if i % 2 == 1:  # 奇数位置放 Mamba
                    self.bottleneck.append(Mamba2DLayer(
                        ch, config.mamba_d_state, config.mamba_d_conv, config.mamba_expand
                    ))
                else:
                    self.bottleneck.append(BasicBlock(ch, ch, dropout=dropout))
        else:
            self.bottleneck = nn.Sequential(*[
                BasicBlock(ch, ch, dropout=dropout) for _ in range(n_bot)
            ])

        # Decoder (5 stages)
        self.decoders = nn.ModuleList()
        for i in range(config.num_encoder_blocks - 1, -1, -1):
            skip_ch = config.base_channels * (2 ** (i + 1))
            out_ch = config.base_channels * (2 ** i)
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch, n_stage, dropout))
            ch = out_ch

        # Output head
        self.final_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        if isinstance(self.bottleneck, nn.ModuleList):
            for layer in self.bottleneck:
                x = layer(x)
        else:
            x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.final_conv(x)  # logits
