# baselines/ditto.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from models import get_model, average_models


class DittoClient:
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
        
        # Global model (for aggregation)
        self.global_model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)
        
        # Personalized model (for evaluation)
        self.personal_model = get_model(
            model_name=config["model"]["name"],
            in_channels=config["input_shape"][0],
            num_classes=config["num_classes"]
        ).to(self.device)

    def receive_global_model(self, global_state_dict: Dict):
        self.global_model.load_state_dict(global_state_dict)
        self.personal_model.load_state_dict(global_state_dict)  # Init personal from global

    def local_train(self) -> Dict:
        # Step 1: Train global model (standard FedAvg step)
        optimizer = optim.SGD(
            self.global_model.parameters(),
            lr=self.config["learning_rate"],
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        self.global_model.train()

        for _ in range(self.config["local_epochs"]):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.global_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Step 2: Train personalized model with proximal term
        lambda_ditto = self.config["ditto"].get("lambda", 0.1)
        opt_personal = optim.SGD(
            self.personal_model.parameters(),
            lr=self.config["learning_rate"],
            momentum=0.9,
            weight_decay=1e-4
        )
        self.personal_model.train()

        global_params = {n: p.clone().detach() for n, p in self.global_model.named_parameters()}
        for _ in range(self.config["local_epochs"]):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                opt_personal.zero_grad()
                output = self.personal_model(data)
                loss_task = criterion(output, target)
                
                # Proximal regularization: ||w_i - w_global||^2
                loss_prox = 0.0
                for n, p in self.personal_model.named_parameters():
                    loss_prox += torch.norm(p - global_params[n]) ** 2
                loss = loss_task + lambda_ditto * loss_prox
                loss.backward()
                opt_personal.step()

        return {
            "client_id": self.client_id,
            "model_state_dict": self.global_model.state_dict(),
            "num_samples": len(self.train_loader.dataset)
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on personalized model!"""
        if self.test_loader is None:
            return {"accuracy": 0.0, "loss": 0.0}
        self.personal_model.eval()
        correct = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.personal_model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        return {
            "accuracy": 100. * correct / len(self.test_loader.dataset),
            "loss": total_loss / len(self.test_loader)
        }


class DittoServer:
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