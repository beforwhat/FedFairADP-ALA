# baselines/adaptive_clipping.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from models import get_model
from core.dp_mechanism import clip_and_noise_client_update  # 复用 DP 核心逻辑


class AdaptiveClippingClient:
    def __init__(
        self,
        client_id: int,
        train_dataset: Subset,
        test_dataset: Optional[Subset],
        config: Dict[str, Any]
    ):
        self.client_id = client_id
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config["batch_size"]) if test_dataset else None
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)
        self.global_snapshot = None

    def receive_global_model(self, global_state_dict: Dict):
        self.model.load_state_dict(global_state_dict)
        self.global_snapshot = global_state_dict  # for clipping reference

    def local_train(self, clip_threshold: float) -> Dict:
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(self.config["local_epochs"]):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                # Optional: clip gradients during training (not required for DP-SGD)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_threshold)
                optimizer.step()

        # Prepare raw update (for DP post-processing)
        update = {
            "client_id": self.client_id,
            "model_state_dict": self.model.state_dict(),
            "num_samples": len(self.train_loader.dataset)
        }

        return update

    def evaluate(self) -> Dict[str, float]:
        if self.test_loader is None:
            return {"accuracy": 0.0, "loss": 0.0}
        self.model.eval()
        correct = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        return {
            "accuracy": 100. * correct / len(self.test_loader.dataset),
            "loss": total_loss / len(self.test_loader)
        }


class AdaptiveClippingServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)
        self.current_round = 0
        
        # DP parameters
        self.initial_clip = config["adaptive_clipping"].get("initial_clip", 1.0)
        self.beta = config["adaptive_clipping"].get("beta", 0.9)  # EMA decay
        self.noise_multiplier = config["dp"]["noise_multiplier"]
        self.target_delta = config["dp"]["target_delta"]
        self.clip_threshold = self.initial_clip

        # Track per-client gradient norms for adaptation
        self.grad_norm_history = []

    def aggregate(self, raw_client_updates: List[Dict]) -> None:
        """
        Step 1: Estimate current round's average gradient norm (pre-clipping)
        Step 2: Update clip threshold via EMA
        Step 3: Apply DP: clip + noise to each client update
        Step 4: FedAvg aggregation
        """
        if not raw_client_updates:
            return

        # --- Step 1 & 2: Adaptive clipping threshold ---
        current_norms = []
        for update in raw_client_updates:
            # Compute ||Δw|| = ||w_local - w_global||
            norm_sq = 0.0
            for k in update["model_state_dict"]:
                diff = update["model_state_dict"][k] - torch.tensor(self.global_model.state_dict()[k])
                norm_sq += torch.sum(diff ** 2).item()
            current_norms.append(norm_sq ** 0.5)
        
        round_avg_norm = sum(current_norms) / len(current_norms)
        self.clip_threshold = self.beta * self.clip_threshold + (1 - self.beta) * round_avg_norm
        self.grad_norm_history.append(round_avg_norm)

        # --- Step 3: Apply DP (clip + noise) ---
        dp_updates = []
        for update in raw_client_updates:
            dp_update = clip_and_noise_client_update(
                client_update=update,
                global_model_state=self.global_model.state_dict(),
                clip_threshold=self.clip_threshold,
                noise_multiplier=self.noise_multiplier,
                device=self.device
            )
            dp_updates.append(dp_update)

        # --- Step 4: Standard FedAvg aggregation ---
        weights = [u["num_samples"] for u in dp_updates]
        total_samples = sum(weights)
        weights = [w / total_samples for w in weights]

        averaged_state = {}
        for key in dp_updates[0]["model_state_dict"]:
            averaged_state[key] = torch.zeros_like(dp_updates[0]["model_state_dict"][key])
            for i, u in enumerate(dp_updates):
                averaged_state[key] += weights[i] * u["model_state_dict"][key]

        self.global_model.load_state_dict(averaged_state)
        self.current_round += 1

    def get_global_model_state(self) -> Dict:
        return self.global_model.state_dict()

    def get_dp_params(self) -> dict:
        """Return current DP settings for privacy accounting"""
        return {
            "clip_threshold": self.clip_threshold,
            "noise_multiplier": self.noise_multiplier,
            "current_round": self.current_round,
            "grad_norm_history": self.grad_norm_history.copy()
        }