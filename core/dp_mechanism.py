# core/dp_mechanism.py

import torch
from typing import Dict, Any
from math import sqrt, log  # 替代 numpy 的 sqrt/log


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