import torch.nn as nn


class LSTM_MSDA(nn.Module):
    """
    LSTM-based probabilistic forecaster for MSDA comparison.
    """
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, quantiles: list):
        super().__init__()
        self.lstm = nn.LSTM(enc_in, hidden_size=64, num_layers=2, batch_first=True)
        self.proj = nn.Linear(64, 1)
        self.quantile_proj = nn.Linear(1, len(quantiles))
        self.pred_len = pred_len

    def forward(self, x):
        # x: [B, L, D]
        out, _ = self.lstm(x)
        last_hidden = out[:, -1:, :].repeat(1, self.pred_len, 1)
        out = self.proj(last_hidden)
        return self.quantile_proj(out), out   # second output used for MMD on features