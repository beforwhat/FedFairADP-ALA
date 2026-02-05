# core/client.py

import copy
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from core.dp_mechanism import SVAdaptiveDPAccountant


class Client:
    def __init__(
        self,
        client_id: int,
        config: Dict[str, Any],
        train_loader,
        test_loader,
        unlabeled_loader=None,   # 新增：无标签数据加载器（可选）
        model_fn=None            # callable to create model
    ):
        self.client_id = client_id
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.unlabeled_loader = unlabeled_loader  # 可为 None
        self.model_fn = model_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_fn().to(self.device)
        self.global_model_snapshot = None
        self.normalized_shapley = 0.0

        # 初始化 DP 引擎
        self.dp_engine = SVAdaptiveDPAccountant(config)

        # 伪标签配置
        pseudo_config = config.get("pseudo_labeling", {})
        self.use_pseudo = pseudo_config.get("enabled", False)
        self.conf_thresh = pseudo_config.get("confidence_threshold", 0.9)
        self.pseudo_weight = pseudo_config.get("loss_weight", 0.5)

    def receive_global_model(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        normalized_shapley: float = 0.0
    ) -> None:
        self.model.load_state_dict(global_state_dict)
        self.global_model_snapshot = copy.deepcopy(global_state_dict)
        self.normalized_shapley = normalized_shapley

    def local_train(self) -> Dict[str, Any]:
        self.model.train()
        lr = self.config["optimizer"]["lr"]
        momentum = self.config["optimizer"].get("momentum", 0.0)
        local_epochs = self.config["local_epochs"]

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum
        )
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        final_gradients = None
        final_grad_norm = 0.0

        for epoch in range(local_epochs):
            # --- 有监督训练 ---
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                # 有监督损失
                output = self.model(data)
                sup_loss = criterion(output, target)

                # 伪标签损失（如果启用且有无标签数据）
                pseudo_loss = 0.0
                if self.use_pseudo and self.unlabeled_loader is not None:
                    try:
                        unlabeled_data = next(unlabeled_iter)
                    except (StopIteration, NameError):
                        unlabeled_iter = iter(self.unlabeled_loader)
                        unlabeled_data = next(unlabeled_iter)
                    unlabeled_data = unlabeled_data.to(self.device)

                    with torch.no_grad():
                        logits_u = self.model(unlabeled_data)
                        probs = F.softmax(logits_u, dim=1)
                        max_probs, pseudo_labels = torch.max(probs, dim=1)
                        mask = max_probs >= self.conf_thresh

                    if mask.any():
                        logits_u_train = self.model(unlabeled_data[mask])
                        pseudo_loss = criterion(logits_u_train, pseudo_labels[mask])
                    else:
                        pseudo_loss = 0.0
                else:
                    pseudo_loss = 0.0

                # 总损失
                loss = sup_loss + self.pseudo_weight * pseudo_loss
                loss.backward()

                # 仅在最后一轮最后一个 batch 记录梯度（用于 DP）
                if epoch == local_epochs - 1 and batch_idx == len(self.train_loader) - 1:
                    final_gradients = {
                        name: param.grad.clone()
                        for name, param in self.model.named_parameters()
                        if param.grad is not None
                    }
                    grad_norm_sq = sum(torch.sum(g ** 2) for g in final_gradients.values())
                    final_grad_norm = torch.sqrt(grad_norm_sq).item()

                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # --- 应用 DP（如果启用）---
        if self.config["dp"]["enabled"]:
            if final_gradients is None:
                raise RuntimeError("No gradients captured for DP.")

            noisy_gradients = self.dp_engine.update_threshold_and_apply_dp(
                local_gradients=final_gradients,
                normalized_shapley=self.normalized_shapley,
                current_grad_norm=final_grad_norm,
                device=self.device
            )

            # 从全局快照 + 带噪梯度重建模型（更稳定）
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in noisy_gradients:
                        # 注意：这里我们假设本地训练只做一次有效更新（简化）
                        # 更严谨做法是记录原始 Δw，但为简化，直接用带噪梯度做一步更新
                        param.copy_(
                            self.global_model_snapshot[name].to(self.device)
                            - self.config["optimizer"]["lr"] * noisy_gradients[name]
                        )

        return {
            "client_id": self.client_id,
            "model_state_dict": self.model.state_dict(),
            "loss": avg_loss,
            "num_samples": len(self.train_loader.dataset),
            "epsilon": self.dp_engine.get_epsilon() if self.config["dp"]["enabled"] else 0.0
        }