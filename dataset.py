"""
dataset.py
----------
Loads the generated CSV, splits it, and wraps it in a PyTorch Dataset.

Also owns the feature scaler, which gets saved alongside the model so
predict.py can apply the exact same normalization at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "hours_until_deadline",
    "task_difficulty",
    "hours_slept",
    "cups_of_coffee",
    "open_browser_tabs",
    "social_media_checks_per_hour",
    "mood_score",
    "previous_completion_rate",
    "is_weekend",
    "noise_level",
]
TARGET_COLUMN = "will_procrastinate"


@dataclass
class Scaler:
    """Minimal standard scaler (mean/std) that is trivially JSON-serializable."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # guard against constant columns
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Scaler":
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Scaler":
        with open(path) as f:
            return cls.from_json(json.load(f))


class ProcrastinationDataset(Dataset):
    """Wraps a feature matrix + label vector as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_and_split(
    csv_path: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Reads the CSV, shuffles, splits into train/val/test, fits a Scaler on the
    train split only (no leakage), and returns three ProcrastinationDataset
    objects plus the fitted Scaler.
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    scaler = Scaler.fit(X_train)

    train_ds = ProcrastinationDataset(scaler.transform(X_train), y_train)
    val_ds = ProcrastinationDataset(scaler.transform(X_val), y_val)
    test_ds = ProcrastinationDataset(scaler.transform(X_test), y_test)

    return train_ds, val_ds, test_ds, scaler
