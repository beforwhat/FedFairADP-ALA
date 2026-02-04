# baselines/dp_fedavg.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from models import get_model, average_models
from core.dp_mechanism import clip_and_noise_client_update  # 复用 DP 逻辑


class DPFedAvgClient:
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
        self.global_snapshot = global_state_dict  # Save for DP clipping

    def local_train(self, clip_threshold: float, noise_multiplier: float) -> Dict:
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
                
                # Clip gradients (optional, but consistent with DP theory)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_threshold)
                optimizer.step()

        update = {
            "client_id": self.client_id,
            "model_state_dict": self.model.state_dict(),
            "num_samples": len(self.train_loader.dataset)
        }

        # Apply DP: clip (local - global) and add noise
        if self.global_snapshot is not None:
            update = clip_and_noise_client_update(
                client_update=update,
                global_model_state=self.global_snapshot,
                clip_threshold=clip_threshold,
                noise_multiplier=noise_multiplier,
                device=self.device
            )
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


class DPFedAvgServer:
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
        self.clip_threshold = config["dp"]["clip_threshold_init"]
        self.noise_multiplier = config["dp"]["noise_multiplier"]

    def aggregate(self, client_updates: List[Dict]) -> None:
        if not client_updates:
            return
        weights = [update["num_samples"] for update in client_updates]
        total_samples = sum(weights)
        weights = [w / total_samples for w in weights]

        client_models = []
        for update in client_updates:
            model = get_model(
                model_name=self.config["model"]["name"],
                in_channels=self.config["input_shape"][0],
                num_classes=self.config["num_classes"]
            )
            model.load_state_dict(update["model_state_dict"])
            client_models.append(model)

        self.global_model = average_models(client_models, weights)
        self.current_round += 1

    def get_global_model_state(self) -> Dict:
        return self.global_model.state_dict()

    def get_dp_params(self) -> tuple:
        return self.clip_threshold, self.noise_multiplier