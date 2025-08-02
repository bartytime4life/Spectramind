'''
SpectraMind V50 – FGS1 Lightcurve Encoder CLI (Ultimate Version)
--------------------------------------------------------------
Encodes raw or calibrated FGS1 lightcurves using Mamba SSM.
Supports output to latent vector, symbolic overlays, SHAP integration, UMAP export, and logging.
'''

import torch
import numpy as np
import argparse
import os
import json
from fgs1_mamba import FGS1MambaEncoder
from pathlib import Path
from v50_debug_log import log_cli_call
from plot_umap_v50 import project_umap_if_requested
from shap_overlay import compute_shap_values


def load_lightcurve(file_path: str) -> torch.Tensor:
    arr = np.load(file_path)  # shape (135000,) or (1, 135000)
    if arr.ndim == 1:
        arr = arr[None, :]  # (1, T)
    return torch.tensor(arr[None, :, :], dtype=torch.float32)  # (B=1, C=1, T)


def save_output(latent: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, latent.detach().cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to FGS1 lightcurve .npy file")
    parser.add_argument("--output", type=str, required=True, help="Path to save encoded latent .npy")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "none"])
    parser.add_argument("--pos-enc", type=str, default=None, choices=["sin", "learned"])
    parser.add_argument("--return-sequence", action="store_true")
    parser.add_argument("--save-umap", type=str, help="Optional .png path for UMAP projection")
    parser.add_argument("--shap-out", type=str, help="Optional .npy path for SHAP vector")
    parser.add_argument("--background", type=str, help="Optional background .npy for SHAP")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    log_cli_call("encode_fgs1_lightcurve.py", vars(args))

    print(f"🔄 Loading FGS1 input: {args.input}")
    x = load_lightcurve(args.input).to(args.device)

    print(f"🧠 Initializing FGS1 Mamba encoder with latent_dim={args.latent_dim}")
    model = FGS1MambaEncoder(
        latent_dim=args.latent_dim,
        pooling=args.pooling,
        positional_encoding=args.pos_enc,
        return_sequence=args.return_sequence
    ).to(args.device)

    print("⚙️  Running encoding...")
    z = model(x)  # (B, D) or (B, T', D)

    print(f"💾 Saving latent vector to: {args.output}")
    save_output(z, args.output)

    # Optional UMAP projection
    if args.save_umap:
        print("🧬 Projecting UMAP...")
        project_umap_if_requested(z, save_path=args.save_umap)

    # Optional SHAP
    if args.shap_out and args.background:
        print("🔍 Computing SHAP explanation...")
        background = load_lightcurve(args.background).to(args.device)
        shap_vals = compute_shap_values(
            model=model,
            input_tensor=x,
            background_data=background,
            target_fn=None,
            use_cpu=(args.device == "cpu"),
            return_numpy=True
        )
        np.save(args.shap_out, shap_vals)
        print(f"🧠 SHAP vector saved to: {args.shap_out}")

    print("✅ Done.")


if __name__ == "__main__":
    main()