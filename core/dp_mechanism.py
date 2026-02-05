# core/dp_mechanism.py

import torch
from typing import Dict, Any
from math import sqrt, log  # 替代 numpy 的 sqrt/log
import torch
from typing import Dict, Optional
from math import log

def compute_dp_noise_scale(
    target_epsilon: float,
    target_delta: float,
    num_rounds: int,
    num_clients_per_round: int,
    total_clients: int,
    noise_multiplier: float = None
) -> float:
    """
    Compute noise scale (sigma) for Gaussian mechanism.
    Fallback analytical approximation if noise_multiplier not provided.
    """
    if noise_multiplier is not None:
        return noise_multiplier

    # Use built-in math instead of numpy
    sampling_rate = num_clients_per_round / total_clients
    sigma = sqrt(2 * log(1.25 / target_delta)) * (
        sqrt(num_rounds) * sampling_rate
    ) / target_epsilon
    return max(sigma, 0.1)


def clip_and_noise_client_update(
    client_update: Dict[str, Any],
    global_model_state: Dict[str, torch.Tensor],
    clip_threshold: float,
    noise_multiplier: float,
    device: torch.device
) -> Dict[str, Any]:
    """
    Clip the model difference (local - global) and add Gaussian noise.
    """
    local_state = client_update["model_state_dict"]
    
    # Compute L2 norm of (local - global)
    with torch.no_grad():
        diff_norm_sq = 0.0
        for name in local_state:
            diff = local_state[name].float() - global_model_state[name].float()
            diff_norm_sq += torch.sum(diff ** 2)
        diff_norm = torch.sqrt(diff_norm_sq)

        scale = min(1.0, clip_threshold / (diff_norm + 1e-8))
        
        # Build clipped difference
        clipped_diff = {}
        for name in local_state:
            diff = local_state[name].float() - global_model_state[name].float()
            clipped_diff[name] = scale * diff

    # Add noise to clipped difference
    noisy_diff = {}
    with torch.no_grad():
        for name, diff in clipped_diff.items():
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * clip_threshold,
                size=diff.size(),
                device=device
            )
            noisy_diff[name] = diff + noise

    # Reconstruct noised local model
    noised_local = {
        name: global_model_state[name] + noisy_diff[name]
        for name in global_model_state
    }

    client_update["model_state_dict"] = noised_local
    return client_update

class SVAdaptiveDPAccountant:
    """
    客户端本地差分隐私引擎，包含：
      - 自适应裁剪阈值（基于 Shapley + 梯度范数变化趋势）
      - 敏感度绑定（L2 裁剪）
      - 高斯机制加噪
      - RDP 隐私预算追踪
    """

    def __init__(self, config: dict):
        dp_config = config.get("dp", {})
        self.enabled = dp_config.get("enabled", False)
        if not self.enabled:
            return

        # 基础参数
        self.T0 = dp_config.get("clip_threshold_init", 1.0)
        self.noise_multiplier = dp_config.get("noise_multiplier", 1.0)
        self.target_delta = dp_config.get("target_delta", 1e-5)
        self.f = dp_config.get("f", 0.8)
        self.u = dp_config.get("u", 0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 客户端状态
        self.current_threshold: float = self.T0
        self.last_grad_norm: Optional[float] = None  # 上一轮梯度范数

        # RDP 隐私会计（α → RDP(α)）
        # 使用标准 α 序列 [1.5, 2.0, ..., 64.0]
        self.rdp_orders = [1.5] + list(range(2, 65))
        self.rdp_cumulative = {alpha: 0.0 for alpha in self.rdp_orders}

    def _compute_gradient_trend(self, current_norm: float) -> float:
        """计算梯度范数相对变化趋势 n_it = (||g^t|| - ||g^{t-1}||) / (||g^{t-1}|| + eps)"""
        if self.last_grad_norm is None:
            self.last_grad_norm = current_norm
            return 0.0  # 第一轮无变化

        if self.last_grad_norm < 1e-8:
            trend = 0.0
        else:
            trend = (current_norm - self.last_grad_norm) / (self.last_grad_norm + 1e-8)

        self.last_grad_norm = current_norm
        return trend

    def update_threshold_and_apply_dp(
        self,
        local_gradients: Dict[str, torch.Tensor],
        normalized_shapley: float = 0.0,
        current_grad_norm: float = 0.0,
        device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """
        执行 DP 流程，并更新隐私消耗。
        返回带噪梯度。
        """
        if not self.enabled:
            return local_gradients

        dev = device or self.device

        # --- 1. 计算梯度趋势 n_it ---
        n_it = self._compute_gradient_trend(current_grad_norm)

        # --- 2. 更新自适应裁剪阈值 ---
        adaptive_factor = 1.0 + self.f * normalized_shapley - self.u * n_it
        self.current_threshold = self.current_threshold * adaptive_factor
        self.current_threshold = max(self.current_threshold, 1e-3)

        # --- 3. 裁剪梯度（绑定敏感度）---
        with torch.no_grad():
            norm_sq = sum(torch.sum(g.float() ** 2) for g in local_gradients.values())
            grad_norm = torch.sqrt(norm_sq).item()
            scale = min(1.0, self.current_threshold / (grad_norm + 1e-8))
            clipped_grads = {
                name: (grad.float() * scale).to(dev)
                for name, grad in local_gradients.items()
            }

        # --- 4. 加噪（高斯机制）---
        sigma = self.noise_multiplier * self.current_threshold
        noisy_grads = {}
        with torch.no_grad():
            for name, grad in clipped_grads.items():
                noise = torch.normal(mean=0.0, std=sigma, size=grad.shape, device=dev)
                noisy_grads[name] = grad + noise

        # --- 5. 更新隐私消耗（RDP）---
        # 高斯机制 RDP: RDP(α) = α / (2σ²)，其中 σ = noise_multiplier（因敏感度已归一化为1）
        # 此处敏感度 = current_threshold，但裁剪后等效敏感度=1，噪声 std = σ * C → 归一化后 σ_eff = noise_multiplier
        for alpha in self.rdp_orders:
            rdp_alpha = alpha / (2 * (self.noise_multiplier ** 2))
            self.rdp_cumulative[alpha] += rdp_alpha

        return noisy_grads

    def get_epsilon(self, target_delta: Optional[float] = None) -> float:
        """
        将累积 RDP 转换为 (ε, δ)-DP。
        使用标准转换公式：
          ε = min_α [ RDP(α) + (log(1/δ) + log(α)) / (α - 1) ]
        """
        if not self.enabled:
            return 0.0

        if target_delta is None:
            target_delta = self.target_delta

        epsilons = []
        for alpha in self.rdp_orders:
            if alpha <= 1:
                continue
            rdpa = self.rdp_cumulative[alpha]
            eps = rdpa + (log(1 / target_delta) + log(alpha)) / (alpha - 1)
            epsilons.append(eps)
        return min(epsilons) if epsilons else float('inf')

    def get_current_threshold(self) -> float:
        return self.current_threshold

    def reset_privacy_accountant(self):
        """重置隐私消耗（如用于新任务）"""
        self.rdp_cumulative = {alpha: 0.0 for alpha in self.rdp_orders}