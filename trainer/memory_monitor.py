"""memory_monitor.py - GPU显存 + 系统内存监控"""
import torch
import psutil
import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
from config import Match3Config


class MemoryMonitor:
    """GPU显存 + 系统内存监控器"""

    def __init__(self, config: Match3Config, writer: Optional[SummaryWriter] = None):
        self.cfg = config
        self.writer = writer
        self.process = psutil.Process(os.getpid())
        self.warning_threshold = getattr(config, "memory_warning_threshold", 0.85)

    def get_gpu_memory(self) -> dict:
        """获取GPU显存使用信息"""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
        }

    def get_system_memory(self) -> dict:
        """获取系统内存使用信息"""
        mem = psutil.virtual_memory()
        proc_mem = self.process.memory_info()

        return {
            "total_gb": mem.total / 1024 / 1024 / 1024,
            "available_gb": mem.available / 1024 / 1024 / 1024,
            "used_gb": mem.used / 1024 / 1024 / 1024,
            "percent": mem.percent,
            "process_rss_mb": proc_mem.rss / 1024 / 1024,
            "process_vms_mb": proc_mem.vms / 1024 / 1024,
        }

    def log_to_tensorboard(self, step: int, prefix: str = "memory"):
        """定期记录显存使用到TensorBoard"""
        if self.writer is None:
            return

        # GPU显存
        gpu_mem = self.get_gpu_memory()
        for k, v in gpu_mem.items():
            self.writer.add_scalar(f"{prefix}/gpu_{k}", v, step)

        # 系统内存
        if getattr(self.cfg, "log_system_memory", True):
            sys_mem = self.get_system_memory()
            self.writer.add_scalar(f"{prefix}/sys_percent", sys_mem["percent"], step)
            self.writer.add_scalar(f"{prefix}/process_rss_mb", sys_mem["process_rss_mb"], step)

    def print_summary(self):
        """打印显存/内存摘要"""
        print("\n--- 显存/内存监控摘要 ---")

        gpu_mem = self.get_gpu_memory()
        if gpu_mem:
            print(f"GPU显存: 已分配={gpu_mem['allocated_mb']:.1f}MB, "
                  f"预留={gpu_mem['reserved_mb']:.1f}MB, "
                  f"峰值={gpu_mem['max_allocated_mb']:.1f}MB")

        sys_mem = self.get_system_memory()
        print(f"系统内存: {sys_mem['used_gb']:.1f}GB / {sys_mem['total_gb']:.1f}GB "
              f"({sys_mem['percent']:.1f}%)")
        print(f"进程RSS: {sys_mem['process_rss_mb']:.1f}MB")

        # 警告
        if sys_mem["percent"] / 100.0 > self.warning_threshold:
            print("⚠️ 系统内存使用超过阈值!")

        print("-" * 30)

    def reset_peak_stats(self):
        """重置峰值统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
