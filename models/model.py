"""model.py - CNN-RNN Bottleneck + Transformer Output U-Net (5层, 每stage 3×ResBlock)

架构:
- 5层 Encoder/Decoder (50x50 -> 25x25 -> 12x12 -> 6x6 -> 3x3 -> 1x1)
- 每个 Encoder/Decoder stage 包含 3个 BasicBlock (ResNet风格)
- Bottleneck: CNN-RNN 混合 (1× BasicBlock + 双向 Row/Col GRU)
- Decoder 末端: Transformer Spatial Block (Self-Attention + FFN)
- 输出: 1×1 Conv logits
- 激活函数使用 Swish (SiLU), 避免 inplace ReLU 导致的 segfault
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Match3Config


class BasicBlock(nn.Module):
    """ResNet BasicBlock: 2x(3x3) + 残差连接 + 可选 Dropout (Swish激活)"""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = (3 // 2) * dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=padding,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=padding,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()  # Swish activation
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


class CNNRNNBottleneck(nn.Module):
    """CNN-RNN 混合 Bottleneck：CNN局部精炼 + 双向Row/Col GRU序列建模

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)
    """

    def __init__(self, channels: int, rnn_hidden_ratio: float = 0.5,
                 num_gru_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.channels = channels

        # CNN 分支: 局部空间精炼
        self.cnn_branch = BasicBlock(channels, channels, dropout=dropout)

        # RNN 分支: 双向 GRU 分别沿行、列方向扫描
        hidden = max(1, int(channels * rnn_hidden_ratio))
        self.hidden = hidden

        # Row-GRU: 沿宽度方向扫描 (每行独立)
        self.row_gru = nn.GRU(
            channels, hidden, num_gru_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_gru_layers > 1 else 0.0
        )
        # Col-GRU: 沿高度方向扫描 (每列独立)
        self.col_gru = nn.GRU(
            channels, hidden, num_gru_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_gru_layers > 1 else 0.0
        )

        # 投影: 2*hidden -> channels
        self.row_proj = nn.Conv2d(hidden * 2, channels, 1, bias=False)
        self.col_proj = nn.Conv2d(hidden * 2, channels, 1, bias=False)

        # 可学习残差缩放
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # --- CNN 分支 ---
        cnn_out = self.cnn_branch(x)

        # --- Row-GRU 分支 ---
        # (B, C, H, W) -> (B*H, W, C)
        x_row = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        row_rnn, _ = self.row_gru(x_row)  # (B*H, W, 2*hidden)
        # -> (B, H, W, 2*hidden) -> (B, 2*hidden, H, W)
        row_rnn = row_rnn.reshape(B, H, W, self.hidden * 2).permute(0, 3, 1, 2)
        row_out = self.row_proj(row_rnn)  # (B, C, H, W)

        # --- Col-GRU 分支 ---
        # (B, C, H, W) -> (B*W, H, C)
        x_col = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        col_rnn, _ = self.col_gru(x_col)  # (B*W, H, 2*hidden)
        # -> (B, W, H, 2*hidden) -> (B, 2*hidden, H, W) 注意要转置回 H,W
        col_rnn = col_rnn.reshape(B, W, H, self.hidden * 2).permute(0, 3, 2, 1)
        col_out = self.col_proj(col_rnn)  # (B, C, H, W)

        # 融合: CNN + (Row + Col) / 2
        fused = cnn_out + (row_out + col_out) * 0.5
        return residual + self.gamma * fused


class TransformerSpatialBlock(nn.Module):
    """2D Transformer Block：将空间特征转为序列做 Self-Attention，再恢复2D

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)

    使用 Pre-LN 结构: LayerNorm -> MSA/FFN -> 残差
    """

    def __init__(self, channels: int, num_heads: int = 8, ffn_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # 确保 channels 可被 num_heads 整除
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        # Pre-LN Multi-Head Self-Attention
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # Pre-LN FFN
        self.norm2 = nn.LayerNorm(channels)
        ffn_hidden = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # 2D -> 序列: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Pre-LN MSA
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_seq = x_seq + self.dropout1(attn_out)

        # Pre-LN FFN
        x_seq = x_seq + self.ffn(self.norm2(x_seq))

        # 序列 -> 2D: (B, H*W, C) -> (B, C, H, W)
        out = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return residual + out


class Match3UNet(nn.Module):
    """CNN-RNN-Transformer U-Net 三消 Pattern 识别模型

    架构:
    - 5层 Encoder (每stage 3× BasicBlock)
    - CNN-RNN Bottleneck (1× BasicBlock + Row/Col Bi-GRU)
    - 5层 Decoder (每stage 3× BasicBlock)
    - Transformer Spatial Block (Self-Attention + FFN)
    - 首层 7x7 大卷积核
    - 输出 logits (无sigmoid, 由loss处理)
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
            nn.SiLU()  # Swish activation
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

        # Bottleneck: ResNet + CNN-RNN 混合
        # 结构: [BasicBlock, CNNRNNBottleneck, CNNRNNBottleneck, BasicBlock]
        # 首尾 ResBlock 精炼特征，中间两层RNN连在一起做序列建模
        use_cnn_rnn = getattr(config, 'use_cnn_rnn', True)
        self.bottleneck = nn.ModuleList()
        rnn_hidden_ratio = getattr(config, 'rnn_hidden_ratio', 0.5)
        num_gru_layers = getattr(config, 'num_gru_layers', 1)
        rnn_dropout = getattr(config, 'rnn_dropout', 0.0)
        for i in range(n_bot):
            if use_cnn_rnn and i in (1, 2):
                # 中间两层: CNN-RNN (两层RNN连在一起)
                self.bottleneck.append(CNNRNNBottleneck(
                    ch, rnn_hidden_ratio, num_gru_layers, rnn_dropout
                ))
            else:
                # 首尾层: 纯 ResBlock (加上ResNet)
                self.bottleneck.append(BasicBlock(ch, ch, dropout=dropout))

        # Decoder (5 stages)
        self.decoders = nn.ModuleList()
        for i in range(config.num_encoder_blocks - 1, -1, -1):
            skip_ch = config.base_channels * (2 ** (i + 1))
            out_ch = config.base_channels * (2 ** i)
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch, n_stage, dropout))
            ch = out_ch

        # Transformer Output Block (可选)
        use_transformer = getattr(config, 'use_transformer_output', True)
        if use_transformer:
            transformer_heads = getattr(config, 'transformer_num_heads', 8)
            transformer_ffn_ratio = getattr(config, 'transformer_ffn_ratio', 4)
            transformer_dropout = getattr(config, 'transformer_dropout', 0.0)
            self.transformer_out = TransformerSpatialBlock(
                ch, transformer_heads, transformer_ffn_ratio, transformer_dropout
            )
        else:
            self.transformer_out = nn.Identity()

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

        x = self.transformer_out(x)
        return self.final_conv(x)  # logits
