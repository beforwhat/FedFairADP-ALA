# core/fairness_selector.py

import random
import numpy as np
from typing import List, Dict


def select_clients_fairly(
    client_ids: List[int],
    participation_counts: Dict[int, int],
    strategy: str = "diversity_aware",
    num_to_select: int = 10,
    diversity_scores: Dict[int, float] = None
) -> List[int]:
    """
    Fair client selection balancing:
      - Participation frequency (avoid over/under-representation)
      - Data diversity (if provided)
    """
    if num_to_select >= len(client_ids):
        return client_ids[:num_to_select]

    if strategy == "uniform":
        return random.sample(client_ids, num_to_select)

    elif strategy == "participation_balanced":
        # Lower participation → higher selection prob
        weights = []
        total_participation = sum(participation_counts.get(cid, 0) for cid in client_ids)
        avg_participation = total_participation / len(client_ids) if client_ids else 1
        
        for cid in client_ids:
            part = participation_counts.get(cid, 0)
            # Inverse weight: under-participated clients get higher chance
            weight = max(1.0, avg_participation / (part + 1e-6))
            weights.append(weight)
        
        return _weighted_sample(client_ids, weights, num_to_select)

    elif strategy == "diversity_aware":
        if diversity_scores is None:
            # Fallback: use participation balance
            return select_clients_fairly(
                client_ids, participation_counts, "participation_balanced", num_to_select
            )
        
        # Combine diversity and participation
        weights = []
        max_div = max(diversity_scores.values()) if diversity_scores else 1.0
        total_participation = sum(participation_counts.get(cid, 0) for cid in client_ids)
        avg_participation = total_participation / len(client_ids) if client_ids else 1
        
        for cid in client_ids:
            div_score = diversity_scores.get(cid, 0.0) / (max_div + 1e-8)
            part = participation_counts.get(cid, 0)
            part_weight = avg_participation / (part + 1e-6)
            # Higher diversity + lower participation → higher weight
            weight = div_score * part_weight
            weights.append(weight)
        
        return _weighted_sample(client_ids, weights, num_to_select)

    else:
        raise ValueError(f"Unknown fairness strategy: {strategy}")


def _weighted_sample(population, weights, k):
    """Weighted random sample without replacement."""
    if k >= len(population):
        return population[:]
    indices = np.random.choice(
        len(population), size=k, replace=False, p=np.array(weights) / np.sum(weights)
    )
    return [population[i] for i in indices]