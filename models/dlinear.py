import torch.nn as nn
from .layers import MovingAvg


class DLinear_MSDA(nn.Module):
    """
    Trend-seasonal decoupled DLinear for probabilistic load forecasting.
    Decomposes input into trend and seasonal components, applies separate
    linear projections, and outputs quantile predictions.
    """
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, quantiles: list):
        super().__init__()
        self.decomposition = MovingAvg(kernel_size=25, stride=1)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(p=0.1)
        self.quantile_proj = nn.Linear(1, len(quantiles))

    def forward(self, x):
        # x: [B, L, D]
        trend_init = self.decomposition(x)              # low-frequency part
        seasonal_init = x - trend_init                  # high-frequency part
        trend_part = self.dropout(
            self.Linear_Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        )
        seasonal_part = self.dropout(
            self.Linear_Seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
        )
        load_out = (trend_part + seasonal_part)[:, :, 0:1]   # only load target
        return self.quantile_proj(load_out), trend_init


class Holistic_DLinear_MSDA(nn.Module):
    """
    Ablation variant: same architecture but WITHOUT trend-seasonal decoupling.
    A single linear layer maps the whole input to future predictions.
    """
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, quantiles: list):
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(p=0.1)
        self.quantile_proj = nn.Linear(1, len(quantiles))

    def forward(self, x):
        out = self.dropout(
            self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        )
        return self.quantile_proj(out[:, :, 0:1]), x   # return original x as "trend" for MMD