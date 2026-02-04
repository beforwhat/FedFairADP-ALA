# core/client.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional
import copy
import numpy as np

from models import get_model


class Client:
    def __init__(
        self,
        client_id: int,
        train_dataset: Subset,
        test_dataset: Optional[Subset],
        config: Dict[str, Any]
    ):
        self.client_id = client_id
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True
        )
        self.test_loader = DataLoader(test_dataset, batch_size=config["batch_size"]) if test_dataset else None
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Local model
        self.model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)
        
        # Store global model state at the start of each round for ALA
        self.global_model_snapshot = None  # Will be set by receive_global_model

    def receive_global_model(self, global_state_dict: Dict):
        """Store a deep copy of the global model for ALA."""
        self.model.load_state_dict(global_state_dict)
        # Save snapshot for later ALA interpolation
        self.global_model_snapshot = copy.deepcopy(global_state_dict)

    def local_train(self, server_clip_threshold: float = float('inf')) -> Dict:
        """
        Perform local training with optional ALA and pseudo-labeling.
        Returns update dict for server.
        """
        # Setup optimizer & scheduler
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            momentum=self.config["optimizer"].get("momentum", 0.9),
            weight_decay=self.config["optimizer"].get("weight_decay", 1e-4)
        )
        
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        total_loss = 0.0
        grad_norms = []

        for epoch in range(self.config["local_epochs"]):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # ===== Pseudo Labeling (if enabled) =====
                if self.config["pseudo_labeling"]["enabled"] and epoch > 0:
                    with torch.no_grad():
                        output = self.model(data)
                        probs = torch.softmax(output, dim=1)
                        max_probs, pseudo_labels = probs.max(dim=1)
                        mask = max_probs > self.config["pseudo_labeling"]["confidence_threshold"]
                        if mask.any():
                            target[mask] = pseudo_labels[mask]

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                # ===== Adaptive Clipping (DP) =====
                if self.config["dp"]["enabled"]:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=server_clip_threshold
                    )

                # Record gradient norm
                total_norm = torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None
                    ]),
                    2
                ).item()
                grad_norms.append(total_norm)

                optimizer.step()
                total_loss += loss.item()

        # ===== Adaptive Local Aggregation (ALA) - FULL IMPLEMENTATION =====
        if self.config["ala"]["enabled"]:
            if self.global_model_snapshot is None:
                raise RuntimeError("Global model snapshot not set. Call receive_global_model first.")
            
            alpha = self.config["ala"]["alpha"]  # Interpolation weight: local = alpha * local + (1-alpha) * global
            
            current_local_state = self.model.state_dict()
            blended_state = {}
            
            with torch.no_grad():
                for key in current_local_state:
                    blended_state[key] = (
                        alpha * current_local_state[key] +
                        (1.0 - alpha) * self.global_model_snapshot[key]
                    )
            
            # Load the blended model back
            self.model.load_state_dict(blended_state)

        avg_loss = total_loss / (len(self.train_loader) * self.config["local_epochs"])
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        return {
            "client_id": self.client_id,
            "model_state_dict": self.model.state_dict(),
            "loss": avg_loss,
            "grad_norm": avg_grad_norm,
            "num_samples": len(self.train_loader.dataset)
        }

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
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return {
            "accuracy": 100. * correct / len(self.test_loader.dataset),
            "loss": total_loss / len(self.test_loader)
        }