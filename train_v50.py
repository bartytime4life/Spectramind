"""
SpectraMind V50 – Symbolic-Aware Training Script (Hydra)
---------------------------------------------------------
Trains full μ + σ prediction pipeline using:
- Symbolic + physics constraints
- GLL loss
- AMP acceleration
- HTML dashboard diagnostics
"""

import os
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime

import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

import hydra
from omegaconf import DictConfig, OmegaConf

from model_v50_ar import SpectraMindModel
from symbolic_loss import compute_total_loss
from calibration_pipeline import run_calibration
from feature_extraction import extract_features
from generate_html_report import generate_html_report


def hash_config(cfg):
    return hashlib.md5(OmegaConf.to_container(cfg, resolve=True).__str__().encode()).hexdigest()


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
            target = torch.tensor(np.load(self.path_dict[pid]["target_mu"]), dtype=torch.float32)
            return features, target
        except Exception as e:
            raise RuntimeError(f"❌ Error loading data for {pid}: {e}")


@hydra.main(config_path="configs", config_name="config_v50.yaml")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed = cfg.training.seed
    torch.manual_seed(seed)

    planet_ids = cfg.data.planet_ids
    metadata = cfg.data.metadata
    paths = cfg.data.paths
    symbolic_cfg = cfg.get("symbolic", {})

    model_cfg = cfg.model_target
    decoder = model_cfg.decoder_type
    lr = cfg.training.learning_rate
    epochs = cfg.training.epochs

    dataset = ArielDataset(planet_ids, metadata, paths)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)

    model = SpectraMindModel(decoder_type=decoder).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    config_hash = hash_config(cfg)
    run_tag = cfg.get("tag", config_hash)
    run_dir = Path(cfg.paths.model_save_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("v50_debug_log.md")
    log_path.write_text(f"\n### Training [{datetime.utcnow().isoformat()}]\nHash: {config_hash}\nDecoder: {decoder}\nTag: {run_tag}\n", append=True)

    log = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, symbolic_terms = 0, {}

        for features, y_true in loader:
            fgs = features["fgs1_sequence"].squeeze(1).cuda()
            airs = features["airs_sequence"].cuda()
            meta = features["metadata"].cuda()
            y_true = y_true.cuda()

            optimizer.zero_grad()
            with autocast():
                y_mu, y_sigma = model(fgs, airs, meta)
                loss, loss_dict = compute_total_loss(y_mu, y_sigma, y_true, symbolic_cfg, epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            for k, v in loss_dict.items():
                symbolic_terms[k] = symbolic_terms.get(k, 0.0) + v

        scheduler.step()
        avg_terms = {k: v / len(loader) for k, v in symbolic_terms.items()}
        avg_terms.update({
            "epoch": epoch,
            "config_hash": config_hash,
            "lr": scheduler.get_last_lr()[0]
        })
        log.append(avg_terms)

        print(f"🧪 Epoch {epoch:02d} | Loss: {total_loss / len(loader):.5f}")
        log_path.write_text(f"Epoch {epoch:02d} | Loss: {total_loss / len(loader):.5f}\n", append=True)

    duration = time.time() - start_time
    log_path.write_text(f"Training completed in {duration:.1f}s\n", append=True)

    with open(run_dir / "training_metrics.json", "w") as f:
        json.dump(log, f, indent=2)

    torch.save(model.state_dict(), run_dir / f"model_{decoder}_e{epochs}.pt")
    print(f"💾 Model saved to {run_dir / f'model_{decoder}_e{epochs}.pt'}")

    generate_html_report(out_path=cfg.tools.html_dashboard)
    print("✅ Training complete and diagnostics report generated.")

if __name__ == "__main__":
    train()