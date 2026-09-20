"""
Baseline MPNN — 8 Configurations
=================================
One architecture, varying only the building blocks:

    h_v^(t+1) = UPDATE(h_v^t, AGGREGATE({ h_u : u ∈ N(v) }))

The 8 configs differ only in:
    - AGGREGATE : sum | mean | max | multi (sum+mean+max)
    - UPDATE    : linear | mlp
    - ACTIVATION: relu | tanh | elu

Config table:
    C1  sum   + linear + relu    (GCN-style)
    C2  mean  + linear + relu
    C3  max   + linear + relu
    C4  sum   + mlp    + relu    (GIN-style)
    C5  mean  + mlp    + relu
    C6  sum   + mlp    + elu
    C7  sum   + mlp    + tanh
    C8  multi + mlp    + relu    (sum+mean+max concatenated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


# ─────────────────────────────────────────────────────────────────────────────
# Single MPNN layer
# ─────────────────────────────────────────────────────────────────────────────

class MPNNLayer(MessagePassing):
    """
    One message-passing layer with configurable aggregation/update/activation.
    Message function is always identity: m_uv = h_u.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 aggr: str = "sum",
                 update: str = "linear",
                 activation: str = "relu"):
        super().__init__(aggr="sum" if aggr == "multi" else aggr)

        self.aggr_type  = aggr
        self.activation = activation

        in_update = in_dim * 3 if aggr == "multi" else in_dim

        if update == "linear":
            self.update_fn = nn.Linear(in_update, out_dim)
        else:
            self.update_fn = nn.Sequential(
                nn.Linear(in_update, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        if self.aggr_type == "multi":
            self.aggr = "sum";  agg_sum  = self.propagate(edge_index, x=x)
            self.aggr = "mean"; agg_mean = self.propagate(edge_index, x=x)
            self.aggr = "max";  agg_max  = self.propagate(edge_index, x=x)
            self.aggr = "sum"   # reset
            agg = torch.cat([agg_sum, agg_mean, agg_max], dim=-1)
        else:
            agg = self.propagate(edge_index, x=x)

        h = self.update_fn(agg)
        h = self.norm(h)
        return self._act(h)

    def message(self, x_j):
        return x_j  # identity: pass neighbour features unchanged

    def _act(self, x):
        if self.activation == "relu":  return F.relu(x)
        if self.activation == "elu":   return F.elu(x)
        if self.activation == "tanh":  return torch.tanh(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class BaselineMPNN(nn.Module):
    """
    Baseline MPNN — one architecture, 8 configurations.

    Args:
        in_dim     : input node feature dim
        hidden_dim : hidden dim (same throughout)
        out_dim    : number of classes
        num_layers : number of MP layers
        aggr       : 'sum' | 'mean' | 'max' | 'multi'
        update     : 'linear' | 'mlp'
        activation : 'relu' | 'elu' | 'tanh'
        dropout    : dropout rate
        task       : 'node' | 'graph'
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, aggr="sum", update="linear",
                 activation="relu", dropout=0.5, task="node"):
        super().__init__()
        self.task    = task
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim,
                      aggr=aggr, update=update, activation=activation)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.config_str = f"aggr={aggr} | update={update} | act={activation}"

    def encode(self, x, edge_index):
        """Node embeddings before classifier head."""
        x = F.relu(self.input_proj(x))
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
        return x

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# 8 named configurations — only the three knobs change
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "C1": dict(aggr="sum",   update="linear", activation="relu"),
    "C2": dict(aggr="mean",  update="linear", activation="relu"),
    "C3": dict(aggr="max",   update="linear", activation="relu"),
    "C4": dict(aggr="sum",   update="mlp",    activation="relu"),
    "C5": dict(aggr="mean",  update="mlp",    activation="relu"),
    "C6": dict(aggr="sum",   update="mlp",    activation="elu"),
    "C7": dict(aggr="sum",   update="mlp",    activation="tanh"),
    "C8": dict(aggr="multi", update="mlp",    activation="relu"),
}

CONFIG_DESCRIPTIONS = {
    "C1": "Sum   + Linear + ReLU   — GCN-style simple baseline",
    "C2": "Mean  + Linear + ReLU   — mean aggregation",
    "C3": "Max   + Linear + ReLU   — max, robust to outlier neighbours",
    "C4": "Sum   + MLP    + ReLU   — GIN-style, deeper update",
    "C5": "Mean  + MLP    + ReLU   — mean with deep update",
    "C6": "Sum   + MLP    + ELU    — smooth non-zero negative response",
    "C7": "Sum   + MLP    + Tanh   — bounded activation",
    "C8": "Multi + MLP    + ReLU   — concat(sum, mean, max), richest signal",
}


def get_model(config_name: str, in_dim: int, hidden_dim: int,
              out_dim: int, **kwargs) -> BaselineMPNN:
    assert config_name in CONFIGS, \
        f"Unknown config '{config_name}'. Choose from: {list(CONFIGS.keys())}"
    return BaselineMPNN(in_dim, hidden_dim, out_dim,
                        **CONFIGS[config_name], **kwargs)