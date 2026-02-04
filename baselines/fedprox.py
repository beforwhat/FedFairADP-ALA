# baselines/fedprox.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from models import get_model, average_models


class FedProxClient:
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
        self.global_snapshot = {k: v.clone().detach() for k, v in global_state_dict.items()}

    def local_train(self) -> Dict:
        mu = self.config["fedprox"].get("mu", 0.01)
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
                loss_task = criterion(output, target)

                # Proximal term: ||w - w_global||^2
                loss_prox = 0.0
                for name, param in self.model.named_parameters():
                    loss_prox += torch.norm(param - self.global_snapshot[name]) ** 2
                loss = loss_task + (mu / 2) * loss_prox

                loss.backward()
                optimizer.step()

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


class FedProxServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)
        self.current_round = 0

    def aggregate(self, client_updates: List[Dict]) -> None:
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