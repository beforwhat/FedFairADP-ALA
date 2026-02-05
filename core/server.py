# core/server.py

import copy
import torch
from typing import List, Dict, Any
from collections import defaultdict

from core.shapley_estimator import estimate_shapley_values
from core.fairness_selector import select_clients_fairly  # 👈 外部引入


def average_models_weighted(
    model_state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """按权重平均多个模型的状态字典"""
    if not model_state_dicts or len(model_state_dicts) != len(weights):
        raise ValueError("Model list and weights must be non-empty and same length.")

    avg_state = {}
    for key in model_state_dicts[0]:
        avg_state[key] = torch.zeros_like(model_state_dicts[0][key], device=device)

    for w, state in zip(weights, model_state_dicts):
        for key in avg_state:
            avg_state[key] += w * state[key].to(device)

    return avg_state


class Server:
    def __init__(
        self,
        config: Dict[str, Any],
        global_model: torch.nn.Module
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_model = copy.deepcopy(global_model).to(self.device)
        self.current_round = 0
        self.client_participation = defaultdict(int)
        self.last_round_normalized_sv = {}  # {client_id: normalized_sv}

    def select_clients(self, all_client_ids: List[int], num_to_select: int) -> List[int]:
        """
        调用外部公平选择器。
        需要传入参与记录以支持公平策略。
        """
        return select_clients_fairly(
            client_ids=all_client_ids,
            participation_counts=dict(self.client_participation),  # 转为普通 dict
            num_to_select=num_to_select,
            strategy=self.config.get("fair_selection", {}).get("strategy", "uniform")
        )

    def aggregate(self, client_updates: List[Dict]) -> None:
        if not client_updates:
            print("Warning: No client updates received.")
            return

        client_ids = [u["client_id"] for u in client_updates]
        model_states = [u["model_state_dict"] for u in client_updates]

        # Step 1: 估计 Shapley 值
        shapley_vals = estimate_shapley_values(
            global_model=self.global_model,
            client_updates=client_updates,
            num_classes=self.config["num_classes"],
            device=self.device,
            mc_samples=self.config.get("shapley", {}).get("mc_samples", 100)
        )

        # Step 2: 归一化 Shapley
        total_sv = sum(shapley_vals[cid] for cid in client_ids)
        if total_sv > 0:
            normalized_sv = {cid: shapley_vals[cid] / total_sv for cid in client_ids}
        else:
            uniform_w = 1.0 / len(client_ids)
            normalized_sv = {cid: uniform_w for cid in client_ids}

        self.last_round_normalized_sv = normalized_sv

        # Step 3: 加权聚合
        weights = [normalized_sv[cid] for cid in client_ids]
        aggregated_state = average_models_weighted(
            model_state_dicts=model_states,
            weights=weights,
            device=self.device
        )
        self.global_model.load_state_dict(aggregated_state)

        # Step 4: 更新参与记录
        for cid in client_ids:
            self.client_participation[cid] += 1
        self.current_round += 1

    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        return self.global_model.state_dict()

    def get_normalized_shapley_for_clients(
        self,
        client_ids: List[int]
    ) -> Dict[int, float]:
        """返回上一轮归一化 Shapley（第0轮返回0.0）"""
        if self.current_round == 0:
            return {cid: 0.0 for cid in client_ids}
        return {
            cid: self.last_round_normalized_sv.get(cid, 0.0)
            for cid in client_ids
        }

    def get_client_participation(self) -> Dict[int, int]:
        return dict(self.client_participation)