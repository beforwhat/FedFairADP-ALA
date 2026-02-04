# core/server.py

import torch
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

from models import get_model, average_models
from core.shapley_estimator import estimate_shapley_values
from core.fairness_selector import select_clients_fairly


class Server:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build global model
        self.global_model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)
        
        self.current_round = 0
        self.client_clip_thresholds = {}  # client_id -> current clip threshold
        self.client_participation = defaultdict(int)  # track selection frequency

    def select_clients(self, all_client_ids: List[int], num_to_select: int) -> List[int]:
        """Fair client selection (diversity + participation-aware)."""
        return select_clients_fairly(
            client_ids=all_client_ids,
            participation_counts=self.client_participation,
            strategy=self.config.get("fair_selection", {}).get("strategy", "uniform"),
            num_to_select=num_to_select
        )

    def aggregate(self, client_updates: List[Dict]) -> None:
        """
        Aggregate client models using Shapley-weighted averaging.
        client_updates: list of {client_id, model_state_dict, loss, grad_norm, ...}
        """
        if not client_updates:
            return

        # Step 1: Estimate Shapley values for selected clients
        shapley_vals = estimate_shapley_values(
            global_model=self.global_model,
            client_updates=client_updates,
            num_classes=self.config["num_classes"],
            device=self.device,
            mc_samples=self.config["shapley"].get("mc_samples", 100)
        )

        # Step 2: Normalize Shapley values to weights
        client_ids = [u["client_id"] for u in client_updates]
        weights = np.array([shapley_vals[cid] for cid in client_ids])
        weights = np.maximum(weights, 1e-8)  # avoid zero weight
        weights /= weights.sum()

        # Step 3: Weighted aggregation
        client_models = []
        for update in client_updates:
            model = get_model(
                model_name=self.config["model"]["name"],
                in_channels=self.config["input_shape"][0],
                num_classes=self.config["num_classes"]
            )
            model.load_state_dict(update["model_state_dict"])
            client_models.append(model)

        self.global_model = average_models(client_models, weights.tolist())

        # Step 4: Update clipping thresholds (for DP clients)
        if self.config["dp"]["enabled"]:
            for update in client_updates:
                cid = update["client_id"]
                # Use Shapley value to adjust sensitivity
                base_threshold = self.config["dp"]["clip_threshold_init"]
                adaptive_factor = 1.0 / (1.0 + shapley_vals[cid])  # higher contribution → lower clip
                self.client_clip_thresholds[cid] = base_threshold * adaptive_factor

        # Update participation counts
        for cid in client_ids:
            self.client_participation[cid] += 1

        self.current_round += 1

    def get_global_model_state(self) -> Dict:
        return self.global_model.state_dict()

    def get_client_clip_threshold(self, client_id: int) -> float:
        """Return adaptive clip threshold for client (used in DP training)."""
        if not self.config["dp"]["enabled"]:
            return float('inf')
        return self.client_clip_thresholds.get(
            client_id, 
            self.config["dp"]["clip_threshold_init"]
        )