"""
train_corel.py – SpectralCOREL GNN Trainer
------------------------------------------
Trains a graph-based uncertainty calibrator (COREL GNN) that refines σ estimates
based on residuals and spectral bin relationships.

Inputs:
- mu_val.pt, sigma_val.pt, y_val.pt: validation predictions
- edge_index.pt: bin connectivity graph

Output:
- Trained GNN saved as models/corel_gnn.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse
from corel import SpectralCOREL


def train_corel_model(mu_val, sigma_val, y_val, edge_index, epochs=50, lr=1e-3):
    """
    Trains SpectralCOREL on (mu, sigma, y) triples.

    Args:
        mu_val (Tensor): [N, 283] predicted mean spectra
        sigma_val (Tensor): [N, 283] predicted stddevs
        y_val (Tensor): [N, 283] ground truth labels
        edge_index (Tensor): [2, E] graph of bin relationships
    """
    model = SpectralCOREL()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    dataset = TensorDataset(mu_val, sigma_val, y_val)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for mu_batch, sigma_batch, y_batch in loader:
            optimizer.zero_grad()
            mu_corr, r_corr = model(mu_batch, sigma_batch, edge_index)

            lower = mu_corr - r_corr
            upper = mu_corr + r_corr
            inside = (y_batch >= lower) & (y_batch <= upper)

            loss = loss_fn(inside.float(), torch.ones_like(inside.float()))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs} - COREL Calibration Loss: {total_loss:.5f}")

    return model


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 COREL model saved to: {path}")


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
    parser.add_argument('--output_path', type=str, default='models/corel_gnn.pt', help='Path to save trained GNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()

    mu_val, sigma_val, y_val, edge_index = load_inputs(args.input_dir)
    model = train_corel_model(mu_val, sigma_val, y_val, edge_index, epochs=args.epochs)
    save_model(model, args.output_path)