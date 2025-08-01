"""
SpectraMind V50 – Symbolic Feedback Retrainer
---------------------------------------------
Retrains decoder using violation heatmaps from constraint_violation_log.json.
Focuses learning on high-violation spectral bins with optional mask weighting,
batch logging, and symbolic loss diagnostics.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model_v50_ar import SpectraMindModel
from symbolic_loss import compute_symbolic_losses
import typer
import os

app = typer.Typer(help="Retrain model using symbolic violation feedback")

class ViolationFocusedDataset(Dataset):
    def __init__(self, submission_file, violation_json):
        df = pd.read_csv(submission_file)
        self.planet_ids = df['planet_id'].tolist()
        self.mu = df[[f"mu_{i}" for i in range(283)]].values
        with open(violation_json) as f:
            self.violations = json.load(f)

        self.mask = np.zeros_like(self.mu)
        for i, pid in enumerate(self.planet_ids):
            if pid in self.violations:
                for bin_idx, score in self.violations[pid].get("bin_scores", {}).items():
                    if 0 <= int(bin_idx) < 283:
                        self.mask[i, int(bin_idx)] = score

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        return torch.tensor(self.mu[idx], dtype=torch.float32), torch.tensor(self.mask[idx], dtype=torch.float32)


@app.command()
def retrain_symbolic(
    submission_csv: Path = Path("submission.csv"),
    violation_log: Path = Path("constraint_violation_log.json"),
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 8,
    symbolic_flags: str = "smoothness,nonnegativity",
    model_out: Path = Path("outputs/symbolic_feedback_model.pt")
):
    """
    Retrains decoder on symbolic failure regions using weighted loss masks.
    """
    os.makedirs(model_out.parent, exist_ok=True)
    flags = {flag.strip(): True for flag in symbolic_flags.split(',') if flag.strip()}
    dataset = ViolationFocusedDataset(submission_csv, violation_log)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpectraMindModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"🔁 Starting symbolic retraining for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for mu, mask in loader:
            mu = mu.requires_grad_()
            loss_dict = compute_symbolic_losses(mu, flags)
            weighted_loss = sum(v * mask.mean() for k, v in loss_dict.items())
            total_loss += weighted_loss.item()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

        print(f"✅ Epoch {epoch+1}/{epochs} | Symbolic Loss = {total_loss:.5f}")

    torch.save(model.state_dict(), model_out)
    print(f"💾 Model saved to {model_out}")

if __name__ == "__main__":
    app()
