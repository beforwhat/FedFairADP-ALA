# utils/misc.py

import random
import numpy as np
import torch
from typing import Any, Dict


def set_seed(seed: int = 42):
    """固定所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
    """
    将嵌套字典展平，用于日志记录
    Example: {'a': {'b': 1}} → {'a/b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(data: Any, device: torch.device):
    """递归将数据移到设备（支持 dict/list/tensor）"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        return data