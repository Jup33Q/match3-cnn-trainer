"""mamba_layer.py - 纯 PyTorch 简化版 2D Mamba 层

不依赖 mamba-ssm，零额外安装。
核心思想：将 2D 特征图通过多种扫描方向展平为 1D 序列，
应用选择性状态空间模型 (S6) 扫描，再合并回 2D。

扫描方向：
- 水平正向 (H, W) -> (H*W,)
- 水平反向
- 垂直正向 (W, H) -> (H*W,)
- 垂直反向
四向结果取平均，融合回 (B, C, H, W)。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScanFn(torch.autograd.Function):
    """选择性扫描的前向 + 反向传播。

    状态方程 (对每个 d_inner 通道独立):
        h_t = A_bar_t * h_{t-1} + (deltaB_t * B_t) * x_t
        y_t = C_t · h_t
    其中 A_bar_t = exp(delta_t * A), deltaB_t = delta_t
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):
        """
        u:     (B, L, d_inner)    输入
        delta: (B, L, d_inner)    时间步长
        A:     (d_inner, d_state) 状态矩阵参数 (已取负)
        B:     (B, L, d_state)    输入相关
        C:     (B, L, d_state)    输出相关
        D:     (d_inner,)         跳跃连接
        """
        B_batch, L, d_inner = u.shape
        d_state = A.shape[1]

        # 离散化: A_bar = exp(delta * A)  [A 已经是负数]
        # delta: (B, L, d_inner) -> (B, L, d_inner, 1)
        # A: (d_inner, d_state) -> (1, 1, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # deltaB: (B, L, d_inner, 1) * (B, L, 1, d_state) = (B, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # 串行扫描 (序列长度通常 <= 2500，串行足够快)
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]  # (B, d_inner, d_state)
            y = (C[:, t].unsqueeze(1) * h).sum(dim=-1)  # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # 跳跃连接 D * u
        if D is not None:
            y = y + u * D.unsqueeze(0).unsqueeze(0)

        ctx.save_for_backward(u, delta, A, B, C, D, deltaA, deltaB_u)
        ctx.L = L
        return y

    @staticmethod
    def backward(ctx, grad_y):
        u, delta, A, B, C, D, deltaA, deltaB_u = ctx.saved_tensors
        L = ctx.L
        B_batch, _, d_inner = u.shape
        d_state = A.shape[1]

        grad_u = torch.zeros_like(u)
        grad_delta = torch.zeros_like(delta)
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        grad_C = torch.zeros_like(C)
        grad_D = torch.zeros_like(D) if D is not None else None

        # 反向传播通过扫描 (reverse)
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        grad_h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        for t in range(L - 1, -1, -1):
            # y_t = sum(C_t * h_t, dim=-1)
            grad_C[:, t] = (h * grad_y[:, t].unsqueeze(-1)).sum(dim=1)  # (B, d_state)
            grad_h += C[:, t].unsqueeze(1) * grad_y[:, t].unsqueeze(-1)  # (B, d_inner, d_state)

            # h_t = deltaA_t * h_{t-1} + deltaB_u_t
            # grad w.r.t. u_t (through deltaB_u = delta * B * u)
            grad_u[:, t] = (delta[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * grad_h).sum(dim=-1)
            if D is not None:
                grad_u[:, t] += D.unsqueeze(0) * grad_y[:, t]
                grad_D += (u[:, t] * grad_y[:, t]).sum(dim=0)

            # grad w.r.t. delta, B
            grad_deltaB_u = grad_h  # (B, d_inner, d_state)
            grad_B[:, t] = (delta[:, t].unsqueeze(-1) * u[:, t].unsqueeze(-1) * grad_deltaB_u).sum(dim=1)  # (B, d_state)
            grad_delta[:, t] = (B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1) * grad_deltaB_u).sum(dim=-1)  # (B, d_inner)

            # grad w.r.t. deltaA
            if t > 0:
                grad_deltaA = h * grad_h  # (B, d_inner, d_state)
                grad_delta[:, t] += (A.unsqueeze(0) * deltaA[:, t] * grad_deltaA).sum(dim=-1)
                grad_A += (delta[:, t].unsqueeze(-1) * deltaA[:, t] * grad_deltaA).sum(dim=0)
                grad_h = deltaA[:, t] * grad_h
                h = deltaA[:, t] * h + deltaB_u[:, t]
            else:
                h = deltaB_u[:, t]

        return grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D


class MambaBlock(nn.Module):
    """单方向 Mamba 块 (序列输入输出)。

    输入:  (B, L, d_model)
    输出:  (B, L, d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)

        # 输入投影: x, z, delta, B, C
        # x, z: d_inner * 2
        # delta: d_inner
        # B, C: d_state * 2
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + self.d_inner + d_state * 2, bias=False)

        # 因果卷积 (深度可分离)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # 可学习参数 A (负指数), shape: (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # 跳跃连接 D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, d_model)
        return: (B, L, d_model)
        """
        B, L, _ = x.shape

        # 输入投影
        proj = self.in_proj(x)  # (B, L, total)
        split1 = self.d_inner * 2
        split2 = split1 + self.d_inner
        split3 = split2 + self.d_state
        split4 = split3 + self.d_state

        xz = proj[..., :split1]
        delta = proj[..., split1:split2]
        B_ssm = proj[..., split2:split3]
        C_ssm = proj[..., split3:split4]

        x_conv, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 因果卷积
        x_conv = x_conv.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # delta 激活
        delta = F.softplus(delta)

        # A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # 选择性扫描
        y = SelectiveScanFn.apply(x_conv, delta, A, B_ssm, C_ssm, self.D)

        # 门控
        z = F.silu(z)
        y = y * z

        # 输出投影
        y = self.out_proj(y)
        return y


class Mamba2DLayer(nn.Module):
    """2D Mamba 层：将 2D 特征通过四向扫描应用 Mamba，再融合。

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)

    扫描方向:
    1. 水平正向 (row-major)
    2. 水平反向
    3. 垂直正向 (column-major)
    4. 垂直反向
    """

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        # 四个方向的 Mamba 块 (参数独立，可以捕获不同方向的空间关系)
        self.mamba_h = MambaBlock(channels, d_state, d_conv, expand)
        self.mamba_h_r = MambaBlock(channels, d_state, d_conv, expand)
        self.mamba_v = MambaBlock(channels, d_state, d_conv, expand)
        self.mamba_v_r = MambaBlock(channels, d_state, d_conv, expand)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        # (B, C, H, W) -> (B, H*W, C)
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_seq = self.norm(x_seq)

        # 1. 水平正向: 每行从左到右
        x_h = x_seq.reshape(B, H, W, C).reshape(B * H, W, C)
        y_h = self.mamba_h(x_h)
        y_h = y_h.reshape(B, H, W, C)

        # 2. 水平反向: 每行从右到左
        x_h_r = torch.flip(x_seq.reshape(B, H, W, C), dims=[2]).reshape(B * H, W, C)
        y_h_r = self.mamba_h_r(x_h_r)
        y_h_r = torch.flip(y_h_r.reshape(B, H, W, C), dims=[2])

        # 3. 垂直正向: 每列从上到下
        x_v = x_seq.reshape(B, H, W, C).permute(0, 2, 1, 3).reshape(B * W, H, C)
        y_v = self.mamba_v(x_v)
        y_v = y_v.reshape(B, W, H, C).permute(0, 2, 1, 3)

        # 4. 垂直反向: 每列从下到上
        x_v_r = torch.flip(x_seq.reshape(B, H, W, C), dims=[1]).permute(0, 2, 1, 3).reshape(B * W, H, C)
        y_v_r = self.mamba_v_r(x_v_r)
        y_v_r = torch.flip(y_v_r.reshape(B, W, H, C).permute(0, 2, 1, 3), dims=[1])

        # 融合四向结果
        y = (y_h + y_h_r + y_v + y_v_r) / 4.0
        y = y.permute(0, 3, 1, 2)  # (B, C, H, W)

        # 残差连接 (可学习缩放)
        return residual + self.gamma * y
