"""
train_corel.py – SpectralCOREL GNN Trainer
------------------------------------------
Trains a graph-based uncertainty calibrator (COREL GNN) that refines σ estimates
based on residuals and spectral bin relationships.

Features:
- Mixed precision (AMP)
- Epoch checkpointing
- CLI compatibility with `spectramind.py corel-train`
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse
from corel import SpectralCOREL
from torch.cuda.amp import autocast, GradScaler


def train_corel_model(mu_val, sigma_val, y_val, edge_index, epochs=50, lr=1e-3, save_every=10, checkpoint_dir="models"):
    """
    Trains SpectralCOREL on (mu, sigma, y) with residual-based interval calibration.

    Args:
        mu_val (Tensor): [N, 283] predicted mean spectra
        sigma_val (Tensor): [N, 283] predicted stddevs
        y_val (Tensor): [N, 283] ground truth labels
        edge_index (Tensor): [2, E] graph of bin relationships
    """
    model = SpectralCOREL()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    scaler = GradScaler()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = TensorDataset(mu_val, sigma_val, y_val)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for mu_batch, sigma_batch, y_batch in loader:
            optimizer.zero_grad()
            with autocast():
                mu_corr, r_corr = model(mu_batch, sigma_batch, edge_index)
                lower = mu_corr - r_corr
                upper = mu_corr + r_corr
                inside = (y_batch >= lower) & (y_batch <= upper)
                loss = loss_fn(inside.float(), torch.ones_like(inside.float()))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs} - COREL Loss: {total_loss:.5f}")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = checkpoint_dir / f"corel_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved to: {ckpt_path}")

    return model


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Final model saved to: {path}")


def load_inputs(input_dir):
    input_dir = Path(input_dir)
    mu = torch.load(input_dir / "mu_val.pt")
    sigma = torch.load(input_dir / "sigma_val.pt")
    y = torch.load(input_dir / "y_val.pt")
    edge = torch.load(input_dir / "edge_index.pt")
    return mu, sigma, y, edge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpectralCOREL GNN uncertainty calibrator")
    parser.add_argument('--input_dir', type=str, default='calibration_data/', help='Directory with mu/sigma/y and edge_index')
    parser.add_argument('--output_path', type=str, default='models/corel_gnn.pt', help='Path to save final trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoints every N epochs')
    args = parser.parse_args()

    mu_val, sigma_val, y_val, edge_index = load_inputs(args.input_dir)
    model = train_corel_model(mu_val, sigma_val, y_val, edge_index, epochs=args.epochs, save_every=args.save_every)
    save_model(model, args.output_path)