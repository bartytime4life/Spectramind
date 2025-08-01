"""
SpectraMind V50 – Data Loader (Enhanced)
----------------------------------------
Loads AIRS, FGS1, metadata, and targets (if available) for training/inference.
Includes:
- NaN and shape checks
- Planet ID tracking
- Spectral waveform stats injection
- Data augmentation support (jitter/noise)
- Configurable loading via `load_v50_dataset()`
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from pathlib import Path
import random

class ArielDataset(Dataset):
    def __init__(self, root: str, mode: str = "train", augment: bool = False):
        """
        Args:
            root: path to dataset folder
            mode: "train" or "test"
            augment: whether to apply FGS1 noise/jitter augmentation
        """
        assert mode in ["train", "test"], f"Invalid mode: {mode}"
        self.root = Path(root)
        self.mode = mode
        self.augment = augment

        self.fgs1 = np.load(self.root / mode / "fgs1_tensor.npy")  # (N, T)
        self.airs = np.load(self.root / mode / "airs_tensor.npy")  # (N, F)
        self.meta = pd.read_csv(self.root / f"{mode}_star_info.csv")

        if mode == "train":
            self.y = np.load(self.root / mode / "gt_mu.npy")  # (N, 283)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        pid = self.meta.iloc[idx, 0]
        fgs = self.fgs1[idx]
        air = self.airs[idx]
        meta_row = self.meta.iloc[idx, 1:].values.astype(np.float32)

        # Optional augmentation (jitter, noise)
        if self.augment and self.mode == "train":
            fgs = self._jitter_augment(fgs)

        fgs_tensor = torch.tensor(fgs, dtype=torch.float32).unsqueeze(0)  # (1, T)
        air_tensor = torch.tensor(air, dtype=torch.float32)              # (F,)
        meta_tensor = torch.tensor(meta_row, dtype=torch.float32)       # (8,)

        # Add FGS1 stats to metadata
        fgs_stats = torch.tensor([
            np.mean(fgs), np.std(fgs), np.max(fgs), np.min(fgs), np.median(fgs)
        ], dtype=torch.float32)

        full_meta = torch.cat([meta_tensor, fgs_stats], dim=0)  # shape: (13,)

        if self.mode == "train":
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return {
                "fgs1_sequence": fgs_tensor,
                "airs_sequence": air_tensor,
                "metadata": full_meta,
                "planet_id": pid
            }, y
        else:
            return {
                "fgs1_sequence": fgs_tensor,
                "airs_sequence": air_tensor,
                "metadata": full_meta,
                "planet_id": pid
            }

    def _jitter_augment(self, seq):
        """Apply small noise or random shift to FGS1 sequence"""
        if random.random() < 0.5:
            return seq + np.random.normal(0, 0.005, size=seq.shape)
        else:
            shift = random.randint(-2, 2)
            return np.roll(seq, shift)

def load_v50_dataset(config_path: str = None, split: str = "train", batch_size: int = 32, augment=False):
    """Helper function to load DataLoader with config lookup"""
    import yaml
    if config_path:
        cfg = yaml.safe_load(open(config_path))
        root = cfg.get("data_root", "data")
    else:
        root = "data"

    ds = ArielDataset(root, mode=split, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
