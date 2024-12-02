from typing import List
import torch.nn as nn
import torch

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# Gradient 시각화
def gradient_p_norm(model: nn.Module, p:int = 2) -> torch.Tensor:
    # 모델 파라미터의 그래디언트 p-노름을 계산
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(p)  # 파라미터 그래디언트의 p-노름 계산
            total_norm += param_norm.item() ** p  # 노름을 p승하여 더함

    total_norm = total_norm ** (1. / p)  # 전체 합의 p제곱근을 취함
    return total_norm

def empirical_fisher_information(model: nn.Module) -> torch.Tensor:
    return gradient_p_norm(model, p = 2)