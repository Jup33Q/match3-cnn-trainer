"""loggers.py - 统一日志接口，支持 TensorBoard 和 JSON 两种后端"""
import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np


def _make_serializable(value: Any) -> Any:
    """将 tensor/numpy 转为 Python 原生类型."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class BaseLogger(ABC):
    """日志抽象基类"""

    @abstractmethod
    def add_scalar(self, tag: str, scalar_value: Any, global_step: Optional[int] = None):
        """记录标量值"""
        pass

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, Any], global_step: Optional[int] = None):
        """批量记录标量 (默认逐条调用 add_scalar)"""
        for tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{tag}", value, global_step)

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TensorBoardLogger(BaseLogger):
    """TensorBoard SummaryWriter 包装"""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag: str, scalar_value: Any, global_step: Optional[int] = None):
        self.writer.add_scalar(tag, _make_serializable(scalar_value), global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, Any], global_step: Optional[int] = None):
        self.writer.add_scalars(main_tag, {_make_serializable(k): _make_serializable(v) for k, v in tag_scalar_dict.items()}, global_step)

    def close(self):
        self.writer.close()


class JsonLogger(BaseLogger):
    """JSON Lines 日志后端 — 不依赖 TensorBoard，可被 Streamlit Dashboard 直接读取"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, "scalars.jsonl")
        self._buffer: list = []
        self._flush_every = 10
        self._counter = 0

    def add_scalar(self, tag: str, scalar_value: Any, global_step: Optional[int] = None):
        record = {
            "step": global_step,
            "tag": tag,
            "value": _make_serializable(scalar_value),
            "wall_time": None,  # 可选
        }
        self._buffer.append(record)
        self._counter += 1
        if self._counter >= self._flush_every:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._buffer.clear()
        self._counter = 0

    def close(self):
        self._flush()


def get_logger(config) -> BaseLogger:
    """根据配置创建对应的 logger"""
    backend = getattr(config, "log_backend", "tensorboard").lower()
    log_dir = getattr(config, "log_dir", "./logs")
    if backend == "json":
        return JsonLogger(log_dir)
    return TensorBoardLogger(log_dir)
