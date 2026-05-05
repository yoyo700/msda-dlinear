import math
import torch
import torch.nn as nn


def dynamic_mmd_weight(epoch: int, total_epochs: int = 30, gamma: float = 10.0, lambda_max: float = 0.003) -> float:
    """
    Sigmoid-schedule for progressive MMD loss weight.
    """
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * epoch / total_epochs)) - 1.0)


class PinballLoss(nn.Module):
    """
    Multi-quantile pinball loss.
    target shape: [B, H, 1] ; preds shape: [B, H, num_quantiles]
    """
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        target = target.expand_as(preds)
        for i, q in enumerate(self.quantiles):
            error = target[:, :, i] - preds[:, :, i]
            loss += torch.max((q - 1) * error, q * error).mean()
        return loss / len(self.quantiles)


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy with multi-scale Gaussian kernels.
    """
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0).flatten(1)
        L2_distance = torch.cdist(total, total, p=2.0) ** 2
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        return sum(torch.exp(-L2_distance / bw) for bw in bandwidth_list)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        return torch.mean(XX + YY - 2 * XY)