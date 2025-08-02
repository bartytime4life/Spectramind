"""
SpectraMind V50 – AIRS Spectral GNN Encoder (All Options)
----------------------------------------------------------
Encodes AIRS spectral traces via edge-aware GNN with positional encoding, symbolic support, and attention tracing.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, NNConv, GINConv, GraphConv,
    global_mean_pool, global_add_pool
)
from typing import Optional, Literal, Tuple


class AIRSSpectralGNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        gnn_type: Literal["gcn", "gat", "sage", "nnconv", "gin", "graphconv"] = "gcn",
        use_edge_attr: bool = False,
        positional_encoding: Optional[Literal["sin", "fourier", "learned"]] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        global_pool: Literal["mean", "sum", "attention"] = "mean",
        return_node_embeddings: bool = False
    ):
        """
        Args:
            in_channels: number of input node features
            hidden_dim: hidden layer width
            latent_dim: output embedding size
            gnn_type: GNN architecture
            use_edge_attr: use edge_attr if available
            positional_encoding: optional spectral bin encoding
            dropout: dropout probability
            use_layer_norm: enable LayerNorm between layers
            global_pool: mean, sum, or attention pooling
            return_node_embeddings: return intermediate embeddings for SHAP/symbolic
        """
        super().__init__()
        self.gnn_type = gnn_type
        self.use_edge_attr = use_edge_attr
        self.return_node_embeddings = return_node_embeddings
        self.positional_encoding = positional_encoding
        self.dropout = dropout

        # Positional encoding
        if positional_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.randn(283, in_channels))

        # GNN layer selector
        if gnn_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, latent_dim)
        elif gnn_type == "gat":
            self.conv1 = GATConv(in_channels, hidden_dim // 2, heads=2)
            self.conv2 = GATConv(hidden_dim, latent_dim // 2, heads=2)
        elif gnn_type == "sage":
            self.conv1 = SAGEConv(in_channels, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, latent_dim)
        elif gnn_type == "gin":
            mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            mlp2 = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)
        elif gnn_type == "graphconv":
            self.conv1 = GraphConv(in_channels, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, latent_dim)
        elif gnn_type == "nnconv":
            edge_net1 = nn.Sequential(nn.Linear(3, hidden_dim * in_channels))
            edge_net2 = nn.Sequential(nn.Linear(3, latent_dim * hidden_dim))
            self.conv1 = NNConv(in_channels, hidden_dim, edge_net1, aggr='mean')
            self.conv2 = NNConv(hidden_dim, latent_dim, edge_net2, aggr='mean')
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.norm1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(latent_dim) if use_layer_norm else nn.Identity()

        # Global pooling
        if global_pool == "mean":
            self.pool = global_mean_pool
        elif global_pool == "sum":
            self.pool = global_add_pool
        elif global_pool == "attention":
            self.pool = global_mean_pool  # placeholder, use attention pool impl later
        else:
            raise ValueError(f"Unsupported pooling: {global_pool}")

    def forward(
        self,
        x: torch.Tensor,                 # (N_nodes, in_channels)
        edge_index: torch.Tensor,       # (2, E)
        batch: torch.Tensor,            # (N_nodes,)
        edge_attr: Optional[torch.Tensor] = None,  # (E, F)
        pos_enc: Optional[torch.Tensor] = None     # (N_nodes,)
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the AIRS GNN encoder.

        Returns:
            - graph_embeddings: (B, latent_dim)
            - node_embeddings (optional): (N, latent_dim) if return_node_embeddings=True
        """
        if self.positional_encoding == "sin" and pos_enc is not None:
            x = x + pos_enc
        elif self.positional_encoding == "learned":
            x = x + self.pos_embed[:x.size(0)]

        conv_kwargs = dict(edge_index=edge_index)
        if self.use_edge_attr and edge_attr is not None:
            conv_kwargs["edge_attr"] = edge_attr

        h = self.conv1(x, **conv_kwargs)
        h = self.norm1(h)
        h = torch.relu(h)
        h = torch.dropout(h, p=self.dropout, train=self.training)

        h = self.conv2(h, **conv_kwargs)
        h = self.norm2(h)
        h = torch.relu(h)

        g = self.pool(h, batch)

        if self.return_node_embeddings:
            return g, h
        return g