"""
SpectraMind V50 – Training Script
----------------------------------
Loads data, extracts features, trains SpectraMindModel using GLL and symbolic losses.
Logs per-epoch symbolic diagnostics to training_metrics.json and v50_debug_log.md.
Supports AMP, config hash tracking, and decoder-specific symbolic tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
import hashlib
import time

from model_v50_ar import SpectraMindModel
from calibration_pipeline import run_calibration
from feature_extraction import extract_features
from generate_html_report import generate_html_report

class ArielDataset(Dataset):
    def __init__(self, planet_ids, metadata_dict, path_dict):
        self.planet_ids = planet_ids
        self.metadata_dict = metadata_dict
        self.path_dict = path_dict

    def __len__(self):
        return len(self.planet_ids)

    def __getitem__(self, idx):
        pid = self.planet_ids[idx]
        calib = run_calibration(pid, self.metadata_dict[pid], self.path_dict[pid])
        features = extract_features(calib, self.metadata_dict[pid])
        target = np.load(self.path_dict[pid]['target_mu'])  # shape (283,)
        return features, torch.tensor(target, dtype=torch.float32)

def gaussian_log_likelihood(y_pred_mu, y_pred_sigma, y_true):
    var = y_pred_sigma ** 2 + 1e-6
    log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((y_true - y_pred_mu) ** 2) / var)
    return -log_prob.sum(dim=-1).mean()

def hash_config(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def train_from_config(cfg):
    planet_ids = cfg["data"]["planet_ids"]
    metadata_dict = cfg["data"]["metadata"]
    path_dict = cfg["data"]["paths"]
    symbolic_config = cfg.get("symbolic", {})
    decoder_type = cfg.get("model", {}).get("decoder_type", "moe")
    lr = cfg.get("training", {}).get("lr", 1e-4)
    max_epochs = cfg.get("training", {}).get("max_epochs", 20)

    dataset = ArielDataset(planet_ids, metadata_dict, path_dict)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SpectraMindModel(decoder_type=decoder_type)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()
    log = []

    config_hash = hash_config(cfg)
    start_time = time.time()

    with open("v50_debug_log.md", "a") as debug:
        debug.write(f"\n### Training Start\nDecoder: {decoder_type}\nHash: {config_hash}\n")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        symbolic_accum = {}

        for batch in loader:
            features, y_true = batch
            fgs = features['fgs1_sequence'].squeeze(1)
            airs = features['airs_sequence']
            meta = features['metadata']

            optimizer.zero_grad()
            with autocast():
                y_mu, y_sigma = model(fgs, airs, meta)
                loss, loss_dict = model.compute_total_loss(y_mu, y_sigma, y_true, symbolic_config, epoch=epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            for k, v in loss_dict.items():
                symbolic_accum[k] = symbolic_accum.get(k, 0.0) + v

        symbolic_avg = {k: v / len(loader) for k, v in symbolic_accum.items()}
        symbolic_avg.update({
            "epoch": epoch,
            "decoder": decoder_type,
            "config_hash": config_hash
        })
        log.append(symbolic_avg)

        print(f"Epoch {epoch:02d} | GLL + Symbolic Loss: {total_loss / len(loader):.5f} | Symbolic Weight: {symbolic_avg.get('symbolic_weight', 0):.3f}")

        with open("v50_debug_log.md", "a") as debug:
            debug.write(f"Epoch {epoch:02d}: Loss = {total_loss / len(loader):.5f}, Decoder = {decoder_type}, Hash = {config_hash}\n")

    duration = time.time() - start_time
    with open("v50_debug_log.md", "a") as debug:
        debug.write(f"Training complete in {duration:.1f} seconds.\n")

    os.makedirs("outputs/training", exist_ok=True)
    with open("outputs/training/training_metrics.json", "w") as f:
        json.dump(log, f, indent=2)

    print("\u2705 Training completed and logged.")
    generate_html_report(decoder_type=decoder_type, config_hash=config_hash)

if __name__ == "__main__":
    train_from_config({})