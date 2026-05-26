"""config.py - 所有可调参数集中管理"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Match3Config:
    """三消 Pattern 识别模型配置 (50x50, Batch=200, BF16, ~8GB显存)"""

    # --- 数据参数 ---
    board_size: int = 50                       # 棋盘尺寸 (50x50)
    num_fruit_types: int = 6                   # 水果种类数
    train_samples: int = 50000                 # 训练样本数
    val_samples: int = 5000                    # 验证样本数

    # --- 数据生成策略 ---
    difficulty_ratio: Tuple[float, float, float] = (0.3, 0.4, 0.3)  # 简单:困难:正例
    min_match_length: int = 3                  # 最小消除长度
    max_match_length: int = 5                  # 最大消除长度 (注入时)

    # --- 模型参数 ---
    initial_kernel_size: int = 7               # 首层大卷积核 (捕获局部连续性)
    base_channels: int = 32                    # 基础通道数
    num_encoder_blocks: int = 5                # Encoder 块数量 (5层深度)
    blocks_per_stage: int = 3                  # 每个Encoder/Decoder stage的ResBlock数
    bottleneck_blocks: int = 4                 # Bottleneck中的ResBlock数
    use_dilation: bool = True                  # 是否使用膨胀卷积
    dilation_rates: List[int] = field(default_factory=lambda: [1, 2, 4, 8])  # 膨胀率序列
    dropout: float = 0.0                       # Dropout率

    # --- Mamba 层参数 ---
    use_mamba: bool = False                    # 是否在Bottleneck中启用Mamba层
    mamba_d_state: int = 16                    # Mamba状态空间维度
    mamba_d_conv: int = 4                      # Mamba因果卷积核大小
    mamba_expand: int = 2                      # Mamba内部扩展因子

    # --- 训练参数 ---
    batch_size: int = 200                      # 每batch样本数
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"               # cosine | plateau | step
    warmup_epochs: int = 5

    # --- 精度设置 ---
    precision: str = "bf16"                    # bf16 | fp16 | fp32
    grad_clip: float = 1.0                     # 梯度裁剪

    # --- 损失函数权重 ---
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    boundary_weight: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # --- 课程学习 ---
    curriculum_enabled: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [10, 25, 50])  # 棋盘递进
    curriculum_epochs_per_stage: int = 20

    # --- 后处理 ---
    mask_threshold: float = 0.5
    enforce_connectivity: bool = True          # 强制连通性约束

    # --- 保存策略 ---
    save_every_batches: int = 10               # 每N个batch记录日志
    save_every_epochs: int = 10                # 每N个epoch保存检查点

    # --- 可视化频率 ---
    visualize_every_epochs: int = 5            # 每N个epoch生成可视化样例 (0=关闭)

    # --- 系统参数 ---
    device: str = "auto"                       # auto | cuda | cpu | mps
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    pin_memory: bool = True
    prefetch_factor: int = 4

    # --- 显存限制 ---
    max_memory_gb: float = 8.0                 # 最大允许显存 (GB)
    memory_warning_threshold: float = 0.85     # 显存告警阈值 (比例)

    def get(self, key, default=None):
        """支持类似dict的get方法"""
        return getattr(self, key, default)
