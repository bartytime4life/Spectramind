"""
SpectraMind V50 – Contrastive Pretraining Pipeline
---------------------------------------------------
Runs contrastive learning using twin augmented AIRS/FGS1 views
with Mamba + GNN encoder and projection head.
Saves checkpoints, logs to v50_debug_log.md, and optionally transitions to MAE.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataloader import load_v50_dataset
from model_v50_ar import SpectraMindModel
from train_mae_v50 import train_mae_from_latents
from pathlib import Path

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=-1)

def contrastive_loss(z1, z2, temperature=0.1):
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    return F.cross_entropy(sim, labels)

def run_contrastive_pretraining(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpectraMindModel().to(device)
    proj_head = ContrastiveHead(in_dim=272, out_dim=cfg["projection_dim"]).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=cfg.get("lr", 1e-4)
    )
    scaler = GradScaler()

    dataset = load_v50_dataset(split="train", batch_size=cfg["batch_size"], augment=True)
    save_dir = Path(cfg.get("save_dir", "outputs/contrastive_pretrain"))
    save_dir.mkdir(parents=True, exist_ok=True)

    debug_log = Path("v50_debug_log.md")
    with open(debug_log, "a") as f:
        f.write(f"\n### Contrastive Pretraining Start\nConfig: {config_path}\n")

    all_latents = []

    print(f"🚀 Starting SpectraMind V50 contrastive pretraining ({cfg['epochs']} epochs)")
    for epoch in range(cfg["epochs"]):
        model.train()
        proj_head.train()
        total_loss = 0.0
        epoch_latents = []

        for batch in tqdm(dataset, desc=f"[Epoch {epoch+1}]"):
            x1, _ = batch
            x2, _ = batch
            fgs1, air1, meta1 = x1['fgs1_sequence'].to(device), x1['airs_sequence'].to(device), x1['metadata'].to(device)
            fgs2, air2, meta2 = x2['fgs1_sequence'].to(device), x2['airs_sequence'].to(device), x2['metadata'].to(device)

            optimizer.zero_grad()
            with autocast():
                z1 = model.encode_latent(fgs1, air1, meta1)
                z2 = model.encode_latent(fgs2, air2, meta2)
                z1_proj = proj_head(z1)
                z2_proj = proj_head(z2)
                loss = contrastive_loss(z1_proj, z2_proj, cfg.get("temperature", 0.1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            epoch_latents.append(z1.detach().cpu())

        avg_loss = total_loss / len(dataset)
        print(f"✅ Epoch {epoch+1}: Contrastive Loss = {avg_loss:.6f}")
        with open(debug_log, "a") as f:
            f.write(f"Epoch {epoch+1}: Contrastive Loss = {avg_loss:.6f}\n")
        torch.save(model.state_dict(), save_dir / f"model_epoch{epoch+1}.pt")
        all_latents.append(torch.cat(epoch_latents, dim=0))

    with open(debug_log, "a") as f:
        f.write(f"Contrastive pretraining complete. Final checkpoint: {save_dir}\n")

    print(f"🎉 Final checkpoints saved to: {save_dir}")

    # Curriculum: Transition to MAE from saved latents
    if cfg.get("auto_mae", False):
        latents_for_mae = torch.cat(all_latents, dim=0)
        with open(debug_log, "a") as f:
            f.write(f"Starting MAE pretraining from contrastive latents...\n")
        train_mae_from_latents(latents_for_mae, cfg.get("mae_config_path", "configs/pretrain/mae.yaml"))
        with open(debug_log, "a") as f:
            f.write(f"MAE pretraining completed.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Contrastive Pretraining for SpectraMind V50")
    parser.add_argument("--config", type=str, default="configs/pretrain/contrastive.json")
    args = parser.parse_args()
    run_contrastive_pretraining(args.config)
