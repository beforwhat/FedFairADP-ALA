# baselines/fedfaira_ala.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from models import get_model
from core.shapley_estimator import estimate_shapley_values
from core.fairness_selector import select_clients_fairly


class FedFairA_ALAClient:
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
        self.global_snapshot = global_state_dict

    def local_train(self) -> Dict:
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
                optimizer.step()

        # ALA: Interpolate local and global
        if self.config["ala"]["enabled"]:
            alpha = self.config["ala"]["alpha"]
            current_state = self.model.state_dict()
            blended = {}
            with torch.no_grad():
                for k in current_state:
                    blended[k] = (
                        alpha * current_state[k] +
                        (1 - alpha) * self.global_snapshot[k]
                    )
            self.model.load_state_dict(blended)

        return {
            "client_id": self.client_id,
            "model_state_dict": self.model.state_dict(),
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
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        return {
            "accuracy": 100. * correct / len(self.test_loader.dataset),
            "loss": total_loss / len(self.test_loader)
        }


class FedFairA_ALAServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)
        self.current_round = 0
        self.participation_counts = {}

    def select_clients(self, all_client_ids: List[int], num_to_select: int) -> List[int]:
        # Fair selection based on participation history
        return select_clients_fairly(
            client_ids=all_client_ids,
            participation_counts=self.participation_counts,
            strategy="participation_balanced",
            num_to_select=num_to_select
        )

    def aggregate(self, client_updates: List[Dict]) -> None:
        if not client_updates:
            return

        # Update participation counts
        for update in client_updates:
            cid = update["client_id"]
            self.participation_counts[cid] = self.participation_counts.get(cid, 0) + 1

        # Shapley-weighted aggregation
        shapley_vals = estimate_shapley_values(
            global_model=self.global_model,
            client_updates=client_updates,
            num_classes=self.config["num_classes"],
            device=self.device,
            mc_samples=self.config["fedfaira_ala"].get("shapley_mc_samples", 50)
        )

        total_shap = sum(shapley_vals.values())
        weights = [
            shapley_vals[update["client_id"]] / total_shap
            for update in client_updates
        ]

        averaged_state = {}
        for key in client_updates[0]["model_state_dict"]:
            averaged_state[key] = torch.zeros_like(client_updates[0]["model_state_dict"][key])
            for i, update in enumerate(client_updates):
                averaged_state[key] += weights[i] * update["model_state_dict"][key]

        self.global_model.load_state_dict(averaged_state)
        self.current_round += 1

    def get_global_model_state(self) -> Dict:
        return self.global_model.state_dict()