# utils/metrics.py

import torch
import numpy as np
from typing import List, Dict, Union, Optional


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """计算分类准确率"""
    return (y_pred == y_true).float().mean().item() * 100.0


def compute_accuracy_per_client(
    client_accuracies: List[float]
) -> Dict[str, float]:
    """计算客户端准确率的统计指标"""
    if not client_accuracies:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(client_accuracies)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def gini_coefficient(values: List[float]) -> float:
    """
    计算基尼系数（衡量不平等程度，0=完全平等，1=完全不平等）
    用于评估客户端性能公平性
    """
    if len(values) == 0:
        return 0.0
    arr = np.sort(np.array(values))
    n = len(arr)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n


def l2_distance(model1: Dict[str, torch.Tensor], model2: Dict[str, torch.Tensor]) -> float:
    """计算两个模型状态字典之间的 L2 距离"""
    total = 0.0
    for key in model1:
        diff = model1[key] - model2[key]
        total += torch.norm(diff, p=2).item() ** 2
    return np.sqrt(total)


def variance_reduction_ratio(
    global_model: Dict[str, torch.Tensor],
    client_models: List[Dict[str, torch.Tensor]]
) -> float:
    """计算客户端模型相对于全局模型的方差缩减比例（FedProx-style）"""
    if not client_models:
        return 0.0
    avg_l2 = np.mean([
        l2_distance(global_model, cm) for cm in client_models
    ])
    return avg_l2


def fairness_metrics(
    accuracies: List[float],
    participation_counts: Optional[List[int]] = None
) -> Dict[str, float]:
    """综合公平性指标"""
    metrics = {}
    metrics["gini"] = gini_coefficient(accuracies)
    metrics["std_dev"] = np.std(accuracies).item()
    metrics["worst_10_percent"] = np.percentile(accuracies, 10)
    if participation_counts is not None:
        # 参与度 vs 性能相关性（理想应低相关）
        corr = np.corrcoef(accuracies, participation_counts)[0, 1] if len(set(participation_counts)) > 1 else 0.0
        metrics["acc_vs_participation_corr"] = float(corr)
    return metrics