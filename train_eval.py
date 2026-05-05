import itertools
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from losses import PinballLoss, MMDLoss, dynamic_mmd_weight


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    source_loader: DataLoader = None,
    use_msda: bool = False,
    device: str = "cpu",
    lr: float = 3e-4,
    epochs: int = 30,
    quantiles: list = [0.1, 0.5, 0.9]
):
    """
    Generic training loop with optional MSDA (MMD on source+target).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_pb = PinballLoss(quantiles).to(device)
    criterion_mmd = MMDLoss().to(device) if use_msda else None

    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0, 0
        lambda_mmd = dynamic_mmd_weight(epoch, epochs) if use_msda else 0.0
        main_loader = source_loader if source_loader is not None else train_loader
        aux_iter = itertools.cycle(train_loader) if source_loader is not None else None

        for main_x, main_y in main_loader:
            steps += 1
            optimizer.zero_grad()

            if source_loader is None:
                # Target-only mode
                p, _ = model(main_x.to(device))
                loss = criterion_pb(p, main_y.to(device))
            else:
                src_x, src_y = main_x.to(device), main_y.to(device)
                tgt_x, tgt_y = next(aux_iter)
                tgt_x, tgt_y = tgt_x.to(device), tgt_y.to(device)

                p_tgt, f_tgt = model(tgt_x)
                p_src, f_src = model(src_x)

                loss = criterion_pb(p_tgt, tgt_y) + criterion_pb(p_src, src_y)

                if use_msda:
                    # Align only minibatch subset to keep batch sizes equal
                    min_size = min(f_src.size(0), f_tgt.size(0))
                    loss += lambda_mmd * criterion_mmd(f_src[:min_size], f_tgt[:min_size])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch[{epoch+1:2d}/{epochs}] | Loss: {total_loss/steps:.4f} | λ: {lambda_mmd:.5f}")

    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler_mu: float,
    scaler_sigma: float,
    device: str = "cpu",
    quantiles: list = [0.1, 0.5, 0.9],
    pred_len: int = 24
):
    """
    Compute Pinball, PICP, PINAW and return one representative day's curves.
    Automatically selects a summer peak day (July) for visualization.
    """
    model.eval()
    all_p, all_t = [], []
    for x, y in test_loader:
        pred, _ = model(x.to(device))
        all_p.append(pred.cpu().numpy())
        all_t.append(y.numpy())

    p_real = np.sort(np.concatenate(all_p, axis=0) * scaler_sigma + scaler_mu, axis=2)
    t_real = np.concatenate(all_t, axis=0) * scaler_sigma + scaler_mu

    # Select a representative summer peak day (July)
    summer_start = 181 * 24
    summer_end = 212 * 24
    summer_daily_max = np.max(t_real[summer_start:summer_end, :, 0], axis=1)
    peak_day_in_summer = summer_start + int(np.argmax(summer_daily_max))

    # Metrics
    picp = np.mean((t_real[:, :, 0] >= p_real[:, :, 0]) & (t_real[:, :, 0] <= p_real[:, :, 2])) * 100
    load_range = np.max(t_real[:, :, 0]) - np.min(t_real[:, :, 0])
    pinaw = np.mean(p_real[:, :, 2] - p_real[:, :, 0]) / load_range if load_range > 0 else 0.0
    pl = PinballLoss(quantiles)(torch.tensor(p_real), torch.tensor(t_real)).item()

    # Extract curves for the selected peak day
    median = p_real[peak_day_in_summer, :, 1]
    lower = p_real[peak_day_in_summer, :, 0]
    upper = p_real[peak_day_in_summer, :, 2]
    true = t_real[peak_day_in_summer, :, 0]

    return round(pl, 2), round(picp, 2), round(pinaw, 3), true, median, lower, upper