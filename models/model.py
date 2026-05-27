"""model.py - Soft-Routed Parallel CNN U-Net (无RNN/Mamba)

架构:
- 5层 Encoder/Decoder (50x50 -> 25x25 -> 12x12 -> 6x6 -> 3x3 -> 1x1)
- 每个 Encoder/Decoder stage 包含 3个 BasicBlock (ResNet风格)
- Bottleneck: Soft-Router + 并行CNN分支 (异步fork/wait) + 加权求和 (router_sum)
- 每个CNN分支内部支持通道扩增/缩减 + 残差 + Swish(SiLU)
- Decoder 末端: Transformer Spatial Block (Self-Attention + FFN)
- 输出: 1×1 Conv logits
- 激活函数使用 Swish (SiLU), 避免 inplace ReLU 导致的 segfault

改进 (v3):
- 移除所有RNN/GRU/Mamba结构
- Bottleneck改为软路由并行CNN分支
- Router生成softmax权重，动态加权各分支输出
- 分支计算通过 torch.jit.fork/wait 异步并行
- 分支内1x1扩增 -> ResBlock -> 1x1缩减，带残差缩放
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Match3Config


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力 (轻量)"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class LayerScale(nn.Module):
    """LayerScale: 可学习残差缩放 (DeiT III / ConvNeXt 风格)

    初始值很小，训练初期残差分支贡献微弱，随训练逐渐增大。
    有效缓解深层网络训练不稳定/梯度爆炸问题。
    支持 (B,C,H,W) 和 (B,C,L) 两种输入格式。
    """

    def __init__(self, channels: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels,), init_value))

    def forward(self, x):
        # x: (B, C, ...) -> gamma: (C,) -> broadcast
        return x * self.gamma.view(1, -1, *([1] * (x.ndim - 2)))


class DropPath(nn.Module):
    """Stochastic Depth / DropPath: 训练时以概率p随机丢弃整个残差分支

    注意: 这是训练稳定性技巧，不同于 Dropout。它帮助深层网络找到更好的优化 basin。
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        # 生成与batch维度相同的随机mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        return x.div(self.keep_prob) * random_tensor


class BasicBlock(nn.Module):
    """ResNet BasicBlock v2: GN + SE + LayerScale + DropPath (SiLU激活)"""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1,
                 dropout: float = 0.0, drop_path: float = 0.0,
                 use_se: bool = True, num_groups: int = 8):
        super().__init__()
        padding = (3 // 2) * dilation
        # GroupNorm: 分割/小batch场景比BatchNorm更稳定
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=padding,
                               dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=padding,
                               dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.layer_scale = LayerScale(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.gn2(self.conv2(x))
        x = self.se(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)
        return self.act(x + residual)


class EncoderBlock(nn.Module):
    """Encoder Stage: N× BasicBlock + MaxPool + Skip Connection"""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 3,
                 use_dilation: bool = False, dilation_rate: int = 1,
                 dropout: float = 0.0, drop_path: float = 0.0,
                 use_se: bool = True, num_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            ich = in_ch if i == 0 else out_ch
            dil = dilation_rate if use_dilation else 1
            self.blocks.append(BasicBlock(ich, out_ch, dil, dropout, drop_path, use_se, num_groups))
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
    """Decoder Stage: Bilinear Upsample + Conv + Concat + N× BasicBlock

    相比 ConvTranspose2d:
    - 避免棋盘格伪影
    - 梯度更稳定
    - 参数量更少
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_blocks: int = 3,
                 dropout: float = 0.0, drop_path: float = 0.0,
                 use_se: bool = True, num_groups: int = 8):
        super().__init__()
        # Bilinear上采样 + 1x1/3x3卷积 替代 ConvTranspose2d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            ich = out_ch + skip_ch if i == 0 else out_ch
            self.blocks.append(BasicBlock(ich, out_ch, dropout=dropout, drop_path=drop_path,
                                          use_se=use_se, num_groups=num_groups))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x)
        return x


class SoftRouter(nn.Module):
    """软路由模块：根据Encoder输出生成各CNN分支的softmax权重

    输入:  (B, C, H, W)
    输出:  (B, num_branches, 1, 1)  未归一化的logits
    """

    def __init__(self, channels: int, num_branches: int, num_groups: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // 4, num_groups), 1, bias=False),
            nn.GroupNorm(num_groups, max(channels // 4, num_groups)),
            nn.SiLU(),
            nn.Conv2d(max(channels // 4, num_groups), num_branches, 1, bias=True),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)


class SoftRouterCNNBranch(nn.Module):
    """软路由CNN分支：1x1扩增 -> N×BasicBlock -> 1x1缩减 + 残差 + Swish

    每个分支独立参数，可配置不同dilation以捕获不同尺度特征。
    输入/输出通道数保持一致，便于加权求和。

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)
    """

    def __init__(self, channels: int, expansion: int = 2, num_blocks: int = 2,
                 dilation: int = 1, dropout: float = 0.0, drop_path: float = 0.0,
                 use_se: bool = True, num_groups: int = 8):
        super().__init__()
        mid_ch = channels * expansion

        # 通道扩增
        self.expand = nn.Sequential(
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.GroupNorm(num_groups, mid_ch),
            nn.SiLU(),
        )

        # 核心处理：N个残差块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                BasicBlock(mid_ch, mid_ch, dilation=dilation, dropout=dropout,
                           drop_path=drop_path, use_se=use_se, num_groups=num_groups)
            )

        # 通道缩减
        self.reduce = nn.Sequential(
            nn.Conv2d(mid_ch, channels, 1, bias=False),
            nn.GroupNorm(num_groups, channels),
        )

        # 分支整体残差缩放
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        x = self.expand(x)
        for block in self.blocks:
            x = block(x)
        x = self.reduce(x)
        # 残差 + Swish
        return F.silu(residual + self.gamma * x)


class RouterSum(nn.Module):
    """路由求和：用softmax权重对各CNN分支输出做加权求和

    输入:
        router_logits: (B, num_branches, 1, 1)
        branch_outs: list of (B, C, H, W)
    输出:
        (B, C, H, W)
    """

    def __init__(self, num_branches: int):
        super().__init__()
        self.num_branches = num_branches

    def forward(self, router_logits, branch_outs):
        if len(branch_outs) != self.num_branches:
            raise ValueError(f"期望 {self.num_branches} 个分支, 得到 {len(branch_outs)}")
        # softmax over branch dimension
        weights = F.softmax(router_logits, dim=1)  # (B, num_branches, 1, 1)
        # 加权求和: weights (B, N, 1, 1) -> broadcast with branch (B, C, H, W)
        out = sum(w.unsqueeze(1) * b for w, b in zip(weights.unbind(dim=1), branch_outs))
        return out


class TransformerSpatialBlock(nn.Module):
    """2D Transformer Block：将空间特征转为序列做 Self-Attention，再恢复2D

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)

    使用 Pre-LN 结构: LayerNorm -> MSA/FFN -> 残差
    添加 LayerScale 稳定训练
    """

    def __init__(self, channels: int, num_heads: int = 8, ffn_ratio: int = 4,
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

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
        self.ls1 = LayerScale(channels)
        self.drop_path1 = DropPath(drop_path)

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
        self.ls2 = LayerScale(channels)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # 2D -> 序列: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Pre-LN MSA
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        # attn_out: (B, HW, C) -> (B, C, HW) -> scale -> (B, HW, C)
        attn_out = self.ls1(attn_out.transpose(1, 2)).transpose(1, 2)
        x_seq = x_seq + self.drop_path1(attn_out)

        # Pre-LN FFN
        ffn_out = self.ffn(self.norm2(x_seq))
        ffn_out = self.ls2(ffn_out.transpose(1, 2)).transpose(1, 2)
        x_seq = x_seq + self.drop_path2(ffn_out)

        # 序列 -> 2D: (B, H*W, C) -> (B, C, H, W)
        out = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return residual + out


class Match3UNet(nn.Module):
    """Soft-Routed Parallel CNN U-Net 三消 Pattern 识别模型 (v3)

    架构:
    - 5层 Encoder (每stage 3× BasicBlock)
    - Soft-Router Bottleneck: Router + 并行CNN分支(fork/wait异步) + 加权求和
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
        num_groups = getattr(config, 'num_groups', 8)

        self.stem = nn.Sequential(
            nn.Conv2d(input_ch, config.base_channels,
                      config.initial_kernel_size,
                      padding=config.initial_kernel_size // 2, bias=False),
            nn.GroupNorm(num_groups, config.base_channels),
            nn.SiLU()
        )

        dropout = getattr(config, 'dropout', 0.0)
        drop_path_rate = getattr(config, 'drop_path_rate', 0.0)
        use_se = getattr(config, 'use_se', True)

        # 计算每层的 drop_path 概率 (线性递增，深层更容易被drop)
        total_depth = config.num_encoder_blocks * n_stage + n_bot + config.num_encoder_blocks * n_stage
        dp_idx = 0

        def get_dp():
            nonlocal dp_idx
            if total_depth <= 1 or drop_path_rate <= 0.0:
                return 0.0
            dp = drop_path_rate * dp_idx / (total_depth - 1)
            dp_idx += 1
            return dp

        # Encoder (5 stages)
        self.encoders = nn.ModuleList()
        ch = config.base_channels
        for i in range(config.num_encoder_blocks):
            next_ch = ch * 2
            dilation = config.dilation_rates[i] if config.use_dilation and i < len(config.dilation_rates) else 1
            stage_dp = [get_dp() for _ in range(n_stage)]
            self.encoders.append(EncoderBlock(ch, next_ch, n_stage, config.use_dilation, dilation,
                                              dropout, max(stage_dp) if stage_dp else 0.0, use_se, num_groups))
            ch = next_ch

        # Bottleneck: Soft-Router + 并行CNN分支
        num_branches = getattr(config, 'num_router_branches', 3)
        router_expansion = getattr(config, 'router_branch_expansion', 2)
        router_blocks = getattr(config, 'router_branch_blocks', 2)
        # 每个分支可配不同dilation，默认 [1, 2, 4]
        default_dilations = [1, 2, 4]
        branch_dilations = getattr(config, 'router_branch_dilations', None)
        if branch_dilations is None:
            branch_dilations = [default_dilations[i % len(default_dilations)] for i in range(num_branches)]
        elif len(branch_dilations) < num_branches:
            branch_dilations = branch_dilations + [default_dilations[i % len(default_dilations)]
                                                    for i in range(len(branch_dilations), num_branches)]

        # Bottleneck预处理块
        self.bottleneck_pre = BasicBlock(ch, ch, dropout=dropout, drop_path=get_dp(),
                                         use_se=use_se, num_groups=num_groups)

        # Router
        self.router = SoftRouter(ch, num_branches, num_groups)

        # 并行CNN分支 (异步fork/wait)
        self.cnn_branches = nn.ModuleList()
        for i in range(num_branches):
            dil = branch_dilations[i]
            dp = get_dp()
            self.cnn_branches.append(
                SoftRouterCNNBranch(ch, expansion=router_expansion, num_blocks=router_blocks,
                                    dilation=dil, dropout=dropout, drop_path=dp,
                                    use_se=use_se, num_groups=num_groups)
            )

        # 路由加权求和
        self.router_sum = RouterSum(num_branches)

        # Bottleneck后处理块
        self.bottleneck_post = BasicBlock(ch, ch, dropout=dropout, drop_path=get_dp(),
                                          use_se=use_se, num_groups=num_groups)

        # Decoder (5 stages)
        self.decoders = nn.ModuleList()
        for i in range(config.num_encoder_blocks - 1, -1, -1):
            skip_ch = config.base_channels * (2 ** (i + 1))
            out_ch = config.base_channels * (2 ** i)
            stage_dp = [get_dp() for _ in range(n_stage)]
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch, n_stage,
                                              dropout, max(stage_dp) if stage_dp else 0.0, use_se, num_groups))
            ch = out_ch

        # Transformer Output Block (可选)
        use_transformer = getattr(config, 'use_transformer_output', True)
        if use_transformer:
            transformer_heads = getattr(config, 'transformer_num_heads', 8)
            transformer_ffn_ratio = getattr(config, 'transformer_ffn_ratio', 4)
            transformer_dropout = getattr(config, 'transformer_dropout', 0.0)
            transformer_drop_path = getattr(config, 'transformer_drop_path', 0.0)
            self.transformer_out = TransformerSpatialBlock(
                ch, transformer_heads, transformer_ffn_ratio, transformer_dropout, transformer_drop_path
            )
        else:
            self.transformer_out = nn.Identity()

        # Output head
        self.final_conv = nn.Conv2d(ch, 1, 1)

        # Xavier (Glorot) Normal 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier Normal 初始化所有可学习参数"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # --- Bottleneck: Soft-Router + 并行CNN分支 ---
        x = self.bottleneck_pre(x)

        # Router生成权重
        router_logits = self.router(x)

        # 异步并行计算各CNN分支 (torch.jit.fork/wait)
        futures = [torch.jit.fork(branch, x) for branch in self.cnn_branches]
        branch_outs = [torch.jit.wait(f) for f in futures]

        # 加权求和
        routed = self.router_sum(router_logits, branch_outs)

        # 残差融合 + 后处理
        x = x + routed
        x = self.bottleneck_post(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        x = self.transformer_out(x)
        return self.final_conv(x)  # logits
