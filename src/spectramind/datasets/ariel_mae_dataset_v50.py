import torch, numpy as np, pandas as pd, os, json
from torch.utils.data import Dataset
from pathlib import Path

class ArielMAEDatasetV50(Dataset):
    def __init__(self, cfg, planet_ids, is_train=True, global_step: int = 0):
        self.cfg = cfg; self.planet_ids = planet_ids; self.is_train = is_train; self.global_step = global_step
        self.mu_df = pd.DataFrame(index=planet_ids, data=np.zeros((len(planet_ids), 283)))
        self.mask_mode = "random"; self.base_mask_ratio = 0.3

    def __len__(self): return len(self.planet_ids)

    def __getitem__(self, idx):
        planet_id = self.planet_ids[idx]
        fgs = torch.zeros(1, 10, 32, 32)  # placeholder shapes
        airs = torch.zeros(1, 10, 32, 356)
        meta = torch.zeros(16)
        mu = torch.zeros(283)
        mask_mu = torch.ones(567)
        mask_sigma = torch.ones(567)
        return {"fgs": fgs, "airs": airs, "meta": meta, "mu": mu, "mask_mu": mask_mu, "mask_sigma": mask_sigma}
