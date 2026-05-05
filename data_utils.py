import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


# ---------- Deterministic seeds ----------
def set_seed(seed: int = 43):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    def seed_worker(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker


# ---------- Dataset ----------
class LoadDataset(Dataset):
    """
    Sliding window dataset for load forecasting.
    Input: multivariate sequences of length seq_len.
    Target: future load values of length pred_len (1-d).
    """
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.x = data
        self.y = data[:, 0:1]   # first column is load
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.x) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s = index
        e = index + self.seq_len
        x = self.x[s:e]
        y = self.y[e:e + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------- Data preprocessing and caching ----------
def build_and_cache_data(
    data_path: str,
    source_zones: list,
    target_zone: str,
    test_hours: int = 8760,
    max_source_hours: int = 8760 * 3,
    few_shot_ratio: float = 0.06,
    cache_file: str = "data_cache.pkl"
):
    """
    Load GEFCom2017, engineer features, normalize, and split source/target data.
    Saves processed data to cache_file.
    """
    print("Processing GEFCom2017 Dataset...")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()
    # Clean time column
    if "time" in df.columns:
        df["time"] = df["time"].str.replace(" CT", "", regex=False)
        df["time"] = pd.to_datetime(df["time"], errors='coerce')
    df = df.dropna(subset=["time"]).sort_values(["zone", "time"]).reset_index(drop=True)

    # Calendar features
    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
    """
    Note: This is a simplified implementation that only distinguishes weekdays from weekends. 
    In practical applications, a dedicated holiday detection function 
    can be designed to replace it with real holiday labels.
    """
    df['is_holiday'] = (df['time'].dt.dayofweek >= 5).astype(int)

    FEATURE_COLS = [
        "load", "temp", "dew_pt",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_holiday"
    ]

    scalers = {}
    processed_source_list = []

    # Source zones: most recent 3 years (excluding the test year)
    for zone in source_zones:
        zone_raw = df[df["zone"] == zone][FEATURE_COLS].values[:-test_hours][-max_source_hours:]
        mu = zone_raw.mean(axis=0)
        sigma = zone_raw.std(axis=0) + 1e-5
        scalers[zone] = {"mu": mu[0], "sigma": sigma[0]}
        processed_source_list.append((zone_raw - mu) / sigma)

    source_data = np.concatenate(processed_source_list, axis=0)

    # Target zone
    target_all = df[df["zone"] == target_zone][FEATURE_COLS].values
    target_test_raw = target_all[-test_hours:]                           # test year
    historical_target = target_all[:-test_hours][-max_source_hours:]     # recent 3 years (excl. test)
    target_train_raw = historical_target[-int(len(historical_target) * few_shot_ratio):]  # tail x%

    mu_tgt = target_train_raw.mean(axis=0)
    sigma_tgt = target_train_raw.std(axis=0) + 1e-5
    scalers[target_zone] = {"mu": mu_tgt[0], "sigma": sigma_tgt[0]}

    cache_data = {
        "FEATURE_COLS": FEATURE_COLS,
        "scalers": scalers,
        "source_data": source_data,
        "target_train": (target_train_raw - mu_tgt) / sigma_tgt,
        "target_test": (target_test_raw - mu_tgt) / sigma_tgt,
    }

    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    return cache_data


def load_data_cache(cache_file: str):
    with open(cache_file, "rb") as f:
        return pickle.load(f)