# core/shapley_estimator.py

import torch
import numpy as np  # ✅ 现在用于 np.random.permutation 和 np.array
from typing import List, Dict, Optional
from collections import defaultdict


def estimate_shapley_values(
    global_model: torch.nn.Module,
    client_updates: List[Dict],
    num_classes: int,
    device: torch.device,
    mc_samples: int = 100,
    eval_batch_size: int = 32,
    public_loader: Optional[List] = None  # 可选：真实验证集
) -> Dict[int, float]:
    """
    Efficient Monte Carlo Shapley estimation.
    Uses random permutations (not full combinations) → O(mc_samples * n), not O(2^n).
    """
    client_ids = [u["client_id"] for u in client_updates]
    n = len(client_ids)
    
    if n == 0:
        return {}
    if n == 1:
        return {client_ids[0]: 1.0}

    # Use real public data if available, else dummy
    if public_loader is not None:
        val_loader = public_loader
    else:
        dummy_x = torch.randn(eval_batch_size, 3, 32, 32)
        dummy_y = torch.randint(0, num_classes, (eval_batch_size,))
        val_loader = [(dummy_x.to(device), dummy_y.to(device))]

    def evaluate_model(model_state):
        model = type(global_model)(
            in_channels=global_model.features[0].in_channels,
            num_classes=num_classes
        ).to(device)
        model.load_state_dict(model_state)
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                loss = torch.nn.functional.cross_entropy(output, y, reduction='sum')
                total_loss += loss.item()
                count += y.size(0)
        return -total_loss / count if count > 0 else 0.0  # Utility = -avg_loss

    base_utility = evaluate_model(global_model.state_dict())
    shapley_sum = defaultdict(float)
    shapley_count = defaultdict(int)

    # Monte Carlo sampling
    for _ in range(mc_samples):
        perm = np.random.permutation(client_ids)  # ✅ 使用 numpy
        coalition_state = global_model.state_dict()
        prev_utility = base_utility

        for i, cid in enumerate(perm):
            # Simple averaging aggregation
            update = next(u for u in client_updates if u["client_id"] == cid)
            new_state = {}
            for key in coalition_state:
                new_state[key] = (
                    (i / (i + 1)) * coalition_state[key] +
                    (1 / (i + 1)) * update["model_state_dict"][key]
                )
            
            curr_utility = evaluate_model(new_state)
            marginal_gain = curr_utility - prev_utility

            shapley_sum[cid] += marginal_gain
            shapley_count[cid] += 1

            coalition_state = new_state
            prev_utility = curr_utility

    # Average and shift to non-negative
    shapley_vals = {
        cid: shapley_sum[cid] / shapley_count[cid]
        for cid in client_ids
    }
    min_val = min(shapley_vals.values())
    shapley_vals = {cid: v - min_val + 1e-8 for cid, v in shapley_vals.items()}
    
    return shapley_vals