"""
SpectraMind V50 – Symbolic-Aware Training Script
--------------------------------------------------
Trains the full SpectraMindModel on μ + σ using symbolic loss, GLL,
AMP acceleration, config hash tracking, cosine LR scheduling,
and diagnostic HTML reporting.
"""

import os
import json
import time
import hashlib
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

from model_v50_ar import SpectraMindModel
from calibration_pipeline import run_calibration
from feature_extraction import extract_features
from generate_html_report import generate_html_report


def hash_config(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

class ArielDataset(Dataset):
    def __init__(self, planet_ids, metadata_dict, path_dict):
        self.planet_ids = planet_ids
        self.metadata_dict = metadata_dict
        self.path_dict = path_dict

    def __len__(self):
        return len(self.planet_ids)

    def __getitem__(self, idx):
        pid = self.planet_ids[idx]
        try:
            calib = run_calibration(pid, self.metadata_dict[pid], self.path_dict[pid])
            features = extract_features(calib, self.metadata_dict[pid])
            target = np.load(self.path_dict[pid]["target_mu"])
            return features, torch.tensor(target, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"❌ Error loading data for {pid}: {e}")

def train_from_config(cfg, dry_run=False, save_model=True, outdir="outputs/training", tag=None):
    planet_ids = cfg["data"]["planet_ids"]
    metadata = cfg["data"]["metadata"]
    paths = cfg["data"]["paths"]
    symbolic = cfg.get("symbolic", {})
    decoder = cfg.get("model", {}).get("decoder_type", "moe")
    lr = cfg.get("training", {}).get("lr", 1e-4)
    epochs = cfg.get("training", {}).get("max_epochs", 20)

    dataset = ArielDataset(planet_ids, metadata, paths)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SpectraMindModel(decoder_type=decoder)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    config_hash = hash_config(cfg)
    run_tag = tag or config_hash
    run_dir = os.path.join(outdir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    log = []
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    with open("v50_debug_log.md", "a") as debug:
        debug.write(f"\n### Training Run [{timestamp}]\nHash: {config_hash}\nDecoder: {decoder}\nTag: {run_tag}\n")

    if dry_run:
        print("🚧 Dry-run mode. Dataset and model initialized.")
        return

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss, symbolic_terms = 0, {}

            for batch in loader:
                features, y_true = batch
                fgs = features["fgs1_sequence"].squeeze(1)
                airs = features["airs_sequence"]
                meta = features["metadata"]

                optimizer.zero_grad()
                with autocast():
                    y_mu, y_sigma = model(fgs, airs, meta)
                    loss, loss_dict = model.compute_total_loss(y_mu, y_sigma, y_true, symbolic, epoch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

                for k, v in loss_dict.items():
                    symbolic_terms[k] = symbolic_terms.get(k, 0.0) + v

            scheduler.step()
            symbolic_avg = {k: v / len(loader) for k, v in symbolic_terms.items()}
            symbolic_avg.update({
                "epoch": epoch,
                "decoder": decoder,
                "config_hash": config_hash,
                "lr": scheduler.get_last_lr()[0]
            })
            log.append(symbolic_avg)

            print(f"Epoch {epoch:02d} | Loss: {epoch_loss / len(loader):.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            with open("v50_debug_log.md", "a") as debug:
                debug.write(f"Epoch {epoch:02d} | Loss = {epoch_loss / len(loader):.5f}\n")

    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user. Saving partial results...")

    duration = time.time() - start_time
    with open("v50_debug_log.md", "a") as debug:
        debug.write(f"Training complete in {duration:.1f} seconds.\n")

    with open(os.path.join(run_dir, "training_metrics.json"), "w") as f:
        json.dump(log, f, indent=2)

    if save_model:
        model_path = os.path.join(run_dir, f"model_{decoder}_e{epochs}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")

    print("✅ Training complete.")
    generate_html_report(decoder_type=decoder, config_hash=config_hash)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--tag", default=None, help="Optional run tag name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--outdir", default="outputs/training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    train_from_config(cfg, dry_run=args.dry_run, save_model=not args.no_save, outdir=args.outdir, tag=args.tag)