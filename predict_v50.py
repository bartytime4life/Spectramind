"""
SpectraMind V50 – Inference Script (Final)
------------------------------------------
Runs full μ + σ inference on test planets using a trained model.
- Uses symbolic-aware calibrated encoder
- Supports temperature scaling, MC dropout
- Outputs μ.pt, σ.pt, and submission.csv
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from model_v50_ar import SpectraMindModel
from calibration_pipeline import run_calibration
from feature_extraction import extract_features
from temperature_scaling import apply_temperature_scaling
from corel_inference import apply_conformal_quantiles


@hydra.main(config_path="configs", config_name="config_v50.yaml")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    test_planets = cfg.data.planet_ids
    metadata_dict = cfg.data.metadata
    path_dict = cfg.data.paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectraMindModel()
    model.load_state_dict(torch.load(cfg.paths.model_save_dir + "/model_final.pt", map_location=device))
    model = model.to(device).eval()

    all_mu, all_sigma = [], []

    for pid in tqdm(test_planets, desc="🔭 Predicting"):
        try:
            calib = run_calibration(pid, metadata_dict[pid], path_dict[pid])
            features = extract_features(calib, metadata_dict[pid])
            fgs = features['fgs1_sequence'].to(device)
            airs = features['airs_sequence'].to(device)
            meta = features['metadata'].to(device)

            with torch.no_grad():
                mu, sigma = model(fgs, airs, meta)
                all_mu.append(mu.squeeze().cpu().numpy())
                all_sigma.append(sigma.squeeze().cpu().numpy())

        except Exception as e:
            print(f"❌ Failed planet {pid}: {e}")
            continue

    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)

    if cfg.inference.temperature_scaling:
        all_sigma = apply_temperature_scaling(all_mu, all_sigma)

    if cfg.inference.quantile_calibration:
        all_sigma = apply_conformal_quantiles(all_mu, all_sigma)

    torch.save(torch.tensor(all_mu), "outputs/mu.pt")
    torch.save(torch.tensor(all_sigma), "outputs/sigma.pt")

    # Write submission
    records = []
    for pid, mu, sigma in zip(test_planets, all_mu, all_sigma):
        row = [pid] + mu.tolist() + sigma.tolist()
        records.append(row)

    columns = ["planet_id"] + [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)]
    df = pd.DataFrame(records, columns=columns)
    out_path = Path(cfg.paths.submission_dir) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ submission.csv written to {out_path}")
    with open("v50_debug_log.md", "a") as log:
        log.write(f"\n✅ μ + σ submission written with {len(df)} planets\n")

if __name__ == "__main__":
    run()