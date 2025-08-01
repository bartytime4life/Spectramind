"""
SpectraMind V50 – MAE Pretraining Script
-----------------------------------------
Pretrains V50 encoders using masked spectrum reconstruction (MAE).
Outputs encoder weights for initialization of downstream supervised training.
"""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F

from model_v50_ar import SpectraMindModel
from dataset_v50 import ArielMAEDatasetV50
from validate_dataset_v50 import validate_dataset

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mae_loss_fn(pred, target, mask):
    loss = (pred - target) ** 2
    return (loss * mask).sum() / (mask.sum() + 1e-8)

def run_mae_pretraining(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    seed_everything(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate_dataset(cfg["paths"]["train_data_dir"], cfg["paths"]["metadata_file"], skip_label=True)

    planet_ids = pd.read_csv(cfg["paths"]["metadata_file"])["planet_id"].tolist()
    dataset = ArielMAEDatasetV50(cfg, planet_ids)
    loader = DataLoader(dataset, batch_size=cfg["mae_pretrain"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])

    model = SpectraMindModel(encoder_only=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["mae_pretrain"]["learning_rate"])
    scaler = GradScaler()

    for epoch in range(cfg["mae_pretrain"]["epochs"]):
        model.train()
        total_loss = 0.0
        for fgs, airs, meta, full_mu, mask in tqdm(loader, desc=f"MAE Epoch {epoch+1}"):
            fgs, airs, meta = fgs.to(device), airs.to(device), meta.to(device)
            full_mu, mask = full_mu.to(device), mask.to(device)

            optimizer.zero_grad()
            with autocast():
                pred_mu = model(fgs, airs, meta)["mu"]
                loss = mae_loss_fn(pred_mu, full_mu, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"\u2705 [MAE] Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

    os.makedirs(cfg["paths"]["model_save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["paths"]["model_save_dir"], "v50_mae_pretrained.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\U0001f389 MAE Pretraining Complete. Saved to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v50_mae.json")
    args = parser.parse_args()
    run_mae_pretraining(args.config)