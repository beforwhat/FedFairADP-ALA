# utils/privacy_checker.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List  
from models import get_model 


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """计算峰值信噪比（PSNR），衡量重建图像质量"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 假设图像已归一化到 [0,1]
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val


class DLGSimulator:
    """
    模拟服务器发起的梯度反演攻击（仅用于评估，非真实攻击）
    参考: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019
    """
    def __init__(
        self,
        model_name: str,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        device: torch.device
    ):
        self.model = get_model(model_name, input_shape[0], num_classes).to(device)
        self.input_shape = input_shape
        self.device = device
        self.original_input: Optional[torch.Tensor] = None  # ✅ 显式声明可选属性

    def simulate_attack(
        self,
        true_gradients: List[torch.Tensor],
        true_labels: torch.Tensor,
        num_trials: int = 3,
        lr: float = 0.1,
        max_iterations: int = 1000
    ) -> Dict[str, float]:
        """
        尝试从梯度中重建原始输入
        返回平均 PSNR（越高表示越容易被重建 → 隐私越差）
        """
        best_psnr = -float('inf')

        for _ in range(num_trials):
            # 初始化随机猜测
            dummy_data = torch.randn(self.input_shape, requires_grad=True, device=self.device)
            dummy_labels = torch.randn(true_labels.shape, requires_grad=True, device=self.device)

            optimizer = optim.LBFGS([dummy_data, dummy_labels], lr=lr)

            history = []
            for it in range(max_iterations):

                def closure():
                    optimizer.zero_grad()
                    self.model.zero_grad()
                    pred = self.model(dummy_data)
                    # 注意：dummy_labels 应为 one-hot 或 logit，这里简化处理
                    dummy_target = dummy_labels.softmax(dim=-1).argmax(dim=-1)
                    dummy_loss = nn.CrossEntropyLoss()(pred, dummy_target)
                    dummy_grads = torch.autograd.grad(
                        dummy_loss, 
                        self.model.parameters(), 
                        create_graph=True
                    )
                    grad_diff = sum(
                        torch.norm(g1 - g2) for g1, g2 in zip(dummy_grads, true_gradients)
                    )
                    grad_diff.backward()
                    history.append(grad_diff.item())
                    return grad_diff

                optimizer.step(closure)
                if len(history) > 5 and abs(history[-1] - history[-5]) < 1e-4:
                    break

            # 如果设置了原始输入（仅用于实验评估），计算 PSNR
            if self.original_input is not None:
                psnr_val = psnr(self.original_input, dummy_data.detach())
                best_psnr = max(best_psnr, psnr_val)

        final_error = history[-1] if history else float('inf')
        result: Dict[str, float] = {
            "dlg_reconstruction_error": final_error,
        }
        if best_psnr != -float('inf'):
            result["psnr"] = best_psnr
        return result

    def set_original_input_for_test(self, x: torch.Tensor):
        """仅用于实验：设置原始输入以计算 PSNR"""
        self.original_input = x.clone().detach().to(self.device)