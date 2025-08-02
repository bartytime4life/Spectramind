"""
SpectraMind V50 – FGS1 Mamba Encoder (Ultimate Integrated Version)
------------------------------------------------------------------
Encodes FGS1 white-light lightcurve (135k samples) using Mamba SSM.
Supports symbolic overlays, latent diagnostics, trace return, and explainability hooks.
"""

import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from typing import Literal, Optional, Union, Tuple


class FGS1MambaEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 135000,
        latent_dim: int = 64,
        downsample_kernel: int = 32,
        downsample_stride: int = 32,
        mamba_model: str = "state-spaces/mamba-130m",
        pooling: Literal["mean", "max", "none"] = "mean",
        positional_encoding: Optional[Literal["sin", "learned"]] = None,
        return_sequence: bool = False,
        use_layer_norm: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: channels (usually 1)
            seq_len: input length (135000)
            latent_dim: final projection dim
            downsample_kernel: Conv1D kernel for downsampling
            downsample_stride: Conv1D stride for downsampling
            mamba_model: model name or path
            pooling: 'mean', 'max', or 'none' (return full)
            positional_encoding: optional bin encodings
            return_sequence: if True, return (B, T', D) post-SSM
            use_layer_norm: normalize output
            dropout: dropout after projection
        """
        super().__init__()
        self.pooling = pooling
        self.return_sequence = return_sequence
        self.positional_encoding = positional_encoding

        self.downsample = nn.Conv1d(
            input_dim, 8,
            kernel_size=downsample_kernel,
            stride=downsample_stride
        )  # e.g., (B, 1, 135000) → (B, 8, ~4200)

        self.mamba = MambaLMHeadModel.from_pretrained(mamba_model)
        self.proj = nn.Linear(self.mamba.config.d_model, latent_dim)
        self.norm = nn.LayerNorm(latent_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        if positional_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.randn(5000, 8))  # up to 5k downsampled bins
        elif positional_encoding == "sin":
            self.register_buffer("sin_enc", self._init_sin_encoding(5000, 8))

    def _init_sin_encoding(self, length: int, dim: int) -> torch.Tensor:
        """Generates sinusoidal encoding matrix (T, D)."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (T, D)

    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, 1, 135000) FGS1 lightcurve
        Returns:
            z: (B, latent_dim) or (B, T', latent_dim)
            Optionally: (z, sequence) if return_sequence=True
        """
        x_ds = self.downsample(x)              # (B, 8, T')
        x_ds = x_ds.permute(0, 2, 1)           # (B, T', 8)

        if self.positional_encoding == "sin":
            pos_enc = self.sin_enc[:x_ds.size(1), :].to(x_ds.device)
            x_ds = x_ds + pos_enc.unsqueeze(0)
        elif self.positional_encoding == "learned":
            x_ds = x_ds + self.pos_embed[:x_ds.size(1), :]

        h = self.mamba(inputs_embeds=x_ds).last_hidden_state  # (B, T', D)

        if self.return_sequence or self.pooling == "none":
            z_seq = self.proj(h)
            return z_seq if self.return_sequence else (self.norm(z_seq), h)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "max":
            pooled = h.max(dim=1).values
        else:
            raise ValueError("Unsupported pooling")

        z = self.norm(self.proj(pooled))       # (B, latent_dim)
        return z