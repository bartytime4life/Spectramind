"""
SpectraMind V50 – Inference Script
----------------------------------
Loads trained model, runs inference on test set, and saves submission.csv
"""

import torch
import numpy as np
import pandas as pd
import os
from core.model_v50_ar import SpectraMindModel
from core.calibration_pipeline import run_calibration
from core.feature_extraction import extract_features


def predict_for_planet(model, planet_id, metadata, paths):
    model.eval()
    with torch.no_grad():
        calib = run_calibration(planet_id, metadata, paths)
        features = extract_features(calib, metadata)
        fgs = features['fgs1_sequence']
        airs = features['airs_sequence']
        meta = features['metadata']
        mu, sigma = model(fgs, airs, meta)
        return mu.squeeze().numpy(), sigma.squeeze().numpy()


def predict():
    test_planets = [...]  # list of planet_ids
    metadata_dict = {...}  # planet_id -> metadata
    path_dict = {...}  # planet_id -> input paths

    model = SpectraMindModel()
    model.load_state_dict(torch.load("checkpoints/model_v50.pt", map_location="cpu"))

    results = []
    for pid in test_planets:
        mu, sigma = predict_for_planet(model, pid, metadata_dict[pid], path_dict[pid])
        row = [pid] + mu.tolist() + sigma.tolist()
        results.append(row)

    columns = ["planet_id"] + [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("submission.csv", index=False)
    print("✅ submission.csv written")


if __name__ == "__main__":
    predict()
