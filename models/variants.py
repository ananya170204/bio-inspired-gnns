"""
Proposed GNN Variants
======================
Three non-baseline architectures, all comparable to BaselineMPNN.

Variant 1: Modular Community GNN
    Inspired by modular brain regions + small-world connectivity.
    Splits hidden dim into K independent modules, each aggregates locally,
    then sparse inter-module communication via a gating mechanism.

Variant 2: Hierarchical Multi-Scale GNN
    Inspired by hierarchical cortical organisation + fractal modularity.
    Low-level layers do local feature extraction, high-level layers
    integrate broader context. Skip connections carry local detail upward.

Variant 3: Parallel Duplication-Based GNN
    Inspired by distributed neural signalling + redundancy in biological systems.
    Duplicates node features across P parallel message-passing channels,
    each with different aggregation, then integrates via learned fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility: basic MP layer (same message fn as baseline)
# ─────────────────────────────────────────────────────────────────────────────

class BasicMPLayer(MessagePassing):
    """Single message-passing layer with one fixed aggregator."""
    def __init__(self, in_dim, out_dim, aggr="sum"):
        super().__init__(aggr=aggr)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        agg = self.propagate(edge_index, x=x)
        return F.relu(self.norm(self.linear(agg)))

    def message(self, x_j):
        return x_j


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1: Modular Community GNN
# ─────────────────────────────────────────────────────────────────────────────

class ModuleBlock(MessagePassing):
    """
    One specialised processing module.
    Performs local message aggregation independently within its feature slice.
    """
    def __init__(self, dim, aggr="sum"):
        super().__init__(aggr=aggr)
        self.linear = nn.Linear(dim, dim)
        self.norm   = nn.BatchNorm1d(dim)

    def forward(self, x, edge_index):
        agg = self.propagate(edge_index, x=x)
        return F.relu(self.norm(self.linear(agg)))

    def message(self, x_j):
        return x_j


class ModularCommunityGNN(nn.Module):
    """
    Variant 1: Modular Community GNN
    
    Architecture:
        - Split hidden_dim into K equal modules (communities)
        - Each module independently aggregates messages in its own subspace
        - Sparse inter-module communication via a learned sparse gate
        - Modules are re-integrated at each layer
    
    Inspiration: modular brain regions, small-world connectivity,
                 efficient resource allocation in neural systems.
    
    Args:
        num_modules : number of independent processing communities (K)
        sparsity    : fraction of inter-module connections to keep (0–1)
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, num_modules=4, sparsity=0.3,
                 dropout=0.5, task="node"):
        super().__init__()
        assert hidden_dim % num_modules == 0, \
            "hidden_dim must be divisible by num_modules"

        self.task        = task
        self.dropout     = dropout
        self.num_modules = num_modules
        self.module_dim  = hidden_dim // num_modules
        self.hidden_dim  = hidden_dim
        self.sparsity    = sparsity

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Per-layer, per-module processing blocks
        self.module_layers = nn.ModuleList([
            nn.ModuleList([
                ModuleBlock(self.module_dim, aggr="sum")
                for _ in range(num_modules)
            ])
            for _ in range(num_layers)
        ])

        # Inter-module sparse communication gate (per layer)
        # Projects full hidden → full hidden but with sparsity mask
        self.inter_module_gates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])

        # Sparse mask: only `sparsity` fraction of inter-module connections active
        # Fixed binary mask (not learned) — inspired by fixed anatomical connectivity
        self._build_sparse_masks(num_layers, hidden_dim, sparsity)

        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _build_sparse_masks(self, num_layers, hidden_dim, sparsity):
        """Build fixed sparse inter-module connection masks."""
        masks = []
        for _ in range(num_layers):
            mask = torch.zeros(hidden_dim, hidden_dim)
            # Allow only cross-module connections (not within-module)
            for i in range(self.num_modules):
                for j in range(self.num_modules):
                    if i != j:  # cross-module only
                        si = i * self.module_dim
                        sj = j * self.module_dim
                        # Randomly keep `sparsity` fraction of cross-module connections
                        submask = (torch.rand(self.module_dim, self.module_dim)
                                   < sparsity).float()
                        mask[si:si+self.module_dim, sj:sj+self.module_dim] = submask
            masks.append(mask)
        # Register as buffers (not parameters — fixed connectivity)
        for i, mask in enumerate(masks):
            self.register_buffer(f"sparse_mask_{i}", mask)

    def encode(self, x, edge_index):
        x = F.relu(self.input_proj(x))

        for layer_idx, module_blocks in enumerate(self.module_layers):
            # 1. Each module processes its own feature slice independently
            module_outputs = []
            for k, block in enumerate(module_blocks):
                start = k * self.module_dim
                end   = start + self.module_dim
                x_k   = x[:, start:end]          # slice for module k
                h_k   = block(x_k, edge_index)   # independent aggregation
                module_outputs.append(h_k)

            # Re-assemble modules
            h = torch.cat(module_outputs, dim=-1)  # [N, hidden_dim]

            # 2. Sparse inter-module communication
            mask  = getattr(self, f"sparse_mask_{layer_idx}")
            gate  = self.inter_module_gates[layer_idx]
            # Apply sparse mask to weight matrix
            sparse_weight = gate.weight * mask
            inter = F.linear(h, sparse_weight)     # sparse cross-module signal
            inter = torch.sigmoid(inter)            # gate: how much to let through

            # 3. Combine intra-module output with sparse inter-module signal
            x = h + 0.1 * inter                    # residual with small inter weight
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2: Hierarchical Multi-Scale GNN
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalMultiScaleGNN(nn.Module):
    """
    Variant 2: Hierarchical Multi-Scale GNN

    Architecture:
        - Low-level layers: local feature extraction (1-hop aggregation)
        - High-level layers: broader context integration (uses output of ALL
          previous layers via skip connections)
        - Skip connections carry fine-grained local detail to higher layers
        - Final representation = concatenation across all scales, projected down

    Inspiration: hierarchical cortical organisation, fractal modularity,
                 multi-scale integration in biological networks.

    Args:
        num_low    : number of low-level (local) layers
        num_high   : number of high-level (global) layers
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_low=2, num_high=2,
                 num_layers=None,  # unused, kept for API compatibility
                 dropout=0.5, task="node"):
        super().__init__()
        self.task       = task
        self.dropout    = dropout
        self.num_low    = num_low
        self.num_high   = num_high
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Low-level layers: local aggregation (sum — preserves count info)
        self.low_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="sum")
            for _ in range(num_low)
        ])

        # High-level layers: broader integration (mean — normalises across scale)
        # Input = hidden_dim (previous high) + hidden_dim (skip from low level)
        self.high_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(num_high)
        ])

        # Skip connection projections: compress low-level outputs for high-level input
        self.skip_projs = nn.ModuleList([
            nn.Linear(hidden_dim * (num_low + 1), hidden_dim)
            for _ in range(num_high)
        ])

        # Final: combine all scale representations
        # We have: input + num_low low outputs + num_high high outputs
        total_scales = 1 + num_low + num_high
        self.scale_fusion = nn.Linear(hidden_dim * total_scales, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x, edge_index):
        x0 = F.relu(self.input_proj(x))
        all_scales = [x0]

        # Low-level: local feature extraction
        h = x0
        for layer in self.low_layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, edge_index)
            all_scales.append(h)

        # High-level: integrate across broader context
        # Each high layer receives current rep + skip from ALL low-level outputs
        h_high = h
        low_context = torch.cat(all_scales, dim=-1)  # all low-level scales

        for i, (layer, skip_proj) in enumerate(
                zip(self.high_layers, self.skip_projs)):
            # Project multi-scale low context to hidden_dim
            skip = F.relu(skip_proj(low_context))
            # Add skip to current high-level representation
            h_high = h_high + skip
            h_high = F.dropout(h_high, p=self.dropout, training=self.training)
            h_high = layer(h_high, edge_index)
            all_scales.append(h_high)

        # Fuse all scales: local detail + global context
        multi_scale = torch.cat(all_scales, dim=-1)
        fused = F.relu(self.scale_fusion(multi_scale))
        return fused

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3: Parallel Duplication-Based GNN
# ─────────────────────────────────────────────────────────────────────────────

class ParallelChannel(MessagePassing):
    """One parallel processing channel with its own aggregator and MLP."""
    def __init__(self, in_dim, out_dim, aggr="sum"):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        agg = self.propagate(edge_index, x=x)
        return F.relu(self.norm(self.mlp(agg)))

    def message(self, x_j):
        return x_j


class ParallelDuplicationGNN(nn.Module):
    """
    Variant 3: Parallel Duplication-Based GNN

    Architecture:
        - Node features duplicated across P parallel channels
        - Each channel uses a DIFFERENT aggregation strategy
          (sum, mean, max — and combinations)
        - Channels process independently through all layers
        - Learned fusion: attention-weighted integration of channel outputs

    Inspiration: distributed neural signalling, redundancy in biological
                 communication pathways, robustness via parallel processing.

    Args:
        num_channels : number of parallel processing streams (P)
                       each gets a different aggregator
    """

    # Fixed aggregator assignment per channel
    CHANNEL_AGGRS = ["sum", "mean", "max", "sum", "mean"]

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, num_channels=3,
                 dropout=0.5, task="node"):
        super().__init__()
        self.task         = task
        self.dropout      = dropout
        self.num_channels = num_channels
        self.hidden_dim   = hidden_dim

        # Each channel gets its own aggregator (cycling through sum/mean/max)
        self.channel_aggrs = [
            self.CHANNEL_AGGRS[i % len(self.CHANNEL_AGGRS)]
            for i in range(num_channels)
        ]

        # Input projection — shared across all channels
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Parallel channel layers: [layer][channel]
        self.channel_layers = nn.ModuleList([
            nn.ModuleList([
                ParallelChannel(hidden_dim, hidden_dim,
                                aggr=self.channel_aggrs[c])
                for c in range(num_channels)
            ])
            for _ in range(num_layers)
        ])

        # Learned attention fusion: weight each channel's contribution
        # Query = mean of all channels, Key = each channel output
        self.fusion_attn = nn.Linear(hidden_dim, num_channels)

        # Final projection after fusion
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x, edge_index):
        x0 = F.relu(self.input_proj(x))

        # Duplicate input across all channels
        channel_outputs = [x0] * self.num_channels  # list of [N, hidden_dim]

        for layer_blocks in self.channel_layers:
            new_outputs = []
            for c, block in enumerate(layer_blocks):
                h_c = F.dropout(channel_outputs[c],
                                p=self.dropout, training=self.training)
                h_c = block(h_c, edge_index)
                new_outputs.append(h_c)
            channel_outputs = new_outputs

        # Learned attention fusion across channels
        # Stack: [N, num_channels, hidden_dim]
        stacked = torch.stack(channel_outputs, dim=1)

        # Compute attention weights from mean channel representation
        mean_rep = stacked.mean(dim=1)                    # [N, hidden_dim]
        attn_logits = self.fusion_attn(mean_rep)          # [N, num_channels]
        attn_weights = F.softmax(attn_logits, dim=-1)     # [N, num_channels]

        # Weighted sum of channels
        # attn_weights: [N, num_channels] → [N, num_channels, 1]
        fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # [N, hidden_dim]
        fused = F.relu(self.fusion_proj(fused))

        return fused

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_CONFIGS = {
    "V1": dict(num_modules=4, sparsity=0.3),
    "V2": dict(num_low=2, num_high=2),
    "V3": dict(num_channels=3),
}

VARIANT_DESCRIPTIONS = {
    "V1": "Modular Community GNN     — K independent modules + sparse inter-module gates",
    "V2": "Hierarchical Multi-Scale  — low-level local + high-level global + skip connections",
    "V3": "Parallel Duplication GNN  — P channels (sum/mean/max) + attention fusion",
}

VARIANT_CLASSES = {
    "V1": ModularCommunityGNN,
    "V2": HierarchicalMultiScaleGNN,
    "V3": ParallelDuplicationGNN,
}


def get_variant(variant_name: str, in_dim: int, hidden_dim: int,
                out_dim: int, **kwargs):
    assert variant_name in VARIANT_CLASSES, \
        f"Unknown variant '{variant_name}'. Choose from: {list(VARIANT_CLASSES.keys())}"
    cls    = VARIANT_CLASSES[variant_name]
    params = {**VARIANT_CONFIGS[variant_name], **kwargs}
    return cls(in_dim, hidden_dim, out_dim, **params)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 4: Hub-Aware GNN
# ─────────────────────────────────────────────────────────────────────────────

class HubAwareMPLayer(MessagePassing):
    """
    Message-passing layer where hub nodes (high degree) send stronger signals.
    Message weight = learned function of sender degree.
    Implements preferential attachment: well-connected nodes influence more.
    """
    def __init__(self, in_dim, out_dim, aggr="sum"):
        super().__init__(aggr=aggr)
        self.linear   = nn.Linear(in_dim, out_dim)
        self.norm     = nn.BatchNorm1d(out_dim)
        # Learned hub scaling: maps log(degree) → scalar weight
        self.hub_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()   # weight in (0, 1) — how much hub influence
        )

    def forward(self, x, edge_index, deg):
        # deg: [N] degree of each node — reshape to [N,1] for PyG indexing
        deg2d = deg.float().unsqueeze(-1)  # [N, 1]
        agg   = self.propagate(edge_index, x=x, deg=deg2d)
        return F.relu(self.norm(self.linear(agg)))

    def message(self, x_j, deg_j):
        # deg_j: [E, 1] degree of the sending node
        log_deg = torch.log1p(deg_j)       # [E, 1]
        weight  = self.hub_gate(log_deg)   # [E, 1]
        return x_j * weight                # scale message by hub importance


class HubAwareGNN(nn.Module):
    """
    Variant 4: Hub-Aware GNN

    Architecture:
        - Computes node degree from edge_index at each forward pass
        - Messages are scaled by a learned function of the sender's degree
        - Hub nodes (high degree) exert stronger influence on neighbours
        - Directly implements preferential attachment / scale-free dynamics

    Inspiration: scale-free organisation (2.3.3), preferential attachment (2.4.2)
                 power-law degree distributions, hub-dominated topologies

    Args:
        None beyond standard — degree computed automatically
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, dropout=0.5, task="node",
                 **kwargs):
        super().__init__()
        self.task    = task
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            HubAwareMPLayer(hidden_dim, hidden_dim, aggr="sum")
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x, edge_index):
        from torch_geometric.utils import degree
        # Compute degree once — used by all layers
        deg = degree(edge_index[0], num_nodes=x.size(0))

        x = F.relu(self.input_proj(x))
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index, deg)
        return x

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 5: Small-World Shortcut GNN
# ─────────────────────────────────────────────────────────────────────────────

class SmallWorldGNN(nn.Module):
    """
    Variant 5: Small-World Shortcut GNN

    Architecture:
        - LOCAL stream: standard message passing on original graph edges
          (high clustering — dense local neighbourhood)
        - GLOBAL stream: learned sparse long-range shortcuts between nodes
          that are NOT direct neighbours (short path lengths)
        - Shortcuts selected by top-k attention scores between node pairs
        - Combine local + global streams via learned gate

    Inspiration: small-world topology (2.3.1, 2.4.1) — high clustering
                 coefficient + short characteristic path length.
                 Economic wiring (2.4.4) — sparse shortcuts balance
                 local efficiency with global integration.

    Args:
        num_shortcuts: number of long-range shortcut edges to add per node (k)
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, num_shortcuts=8,
                 dropout=0.5, task="node", **kwargs):
        super().__init__()
        self.task          = task
        self.dropout       = dropout
        self.num_shortcuts = num_shortcuts

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Local stream: standard MP on original graph
        self.local_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(num_layers)
        ])

        # Global stream: MP on shortcut edges
        self.global_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(num_layers)
        ])

        # Shortcut attention: scores pairs of nodes for shortcut worthiness
        self.shortcut_attn = nn.Linear(hidden_dim * 2, 1)

        # Gate: how much local vs global to combine
        self.combine_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _build_shortcuts(self, x, edge_index, num_nodes):
        """
        Build sparse long-range shortcut edges using attention scores.
        Selects top-k non-neighbour pairs for each node.
        For efficiency, samples a candidate set rather than all pairs.
        """
        k = min(self.num_shortcuts, num_nodes - 1)

        # Sample random candidate pairs (efficient approximation)
        num_candidates = min(num_nodes * k * 2, 2000)
        src = torch.randint(0, num_nodes, (num_candidates,), device=x.device)
        dst = torch.randint(0, num_nodes, (num_candidates,), device=x.device)

        # Remove self-loops
        mask = src != dst
        src, dst = src[mask], dst[mask]

        if src.size(0) == 0:
            # Fallback: return empty edge index
            return torch.zeros(2, 0, dtype=torch.long, device=x.device)

        # Score each candidate pair by node feature similarity
        pair_feat = torch.cat([x[src], x[dst]], dim=-1)  # [E_cand, 2*H]
        scores    = self.shortcut_attn(pair_feat).squeeze(-1)  # [E_cand]

        # Keep top-k*N shortcuts overall
        num_keep = min(k * num_nodes, src.size(0))
        _, top_idx = scores.topk(num_keep)
        shortcut_index = torch.stack([src[top_idx], dst[top_idx]], dim=0)

        return shortcut_index

    def encode(self, x, edge_index):
        num_nodes = x.size(0)
        x = F.relu(self.input_proj(x))

        for local_layer, global_layer in zip(self.local_layers, self.global_layers):
            # LOCAL: aggregate from structural neighbours
            x_local = F.dropout(x, p=self.dropout, training=self.training)
            x_local = local_layer(x_local, edge_index)

            # GLOBAL: build shortcuts from current embeddings, aggregate
            with torch.no_grad() if not self.training else torch.enable_grad():
                shortcut_index = self._build_shortcuts(x, edge_index, num_nodes)

            x_global = F.dropout(x, p=self.dropout, training=self.training)
            if shortcut_index.size(1) > 0:
                x_global = global_layer(x_global, shortcut_index)
            else:
                x_global = x_local  # fallback if no shortcuts

            # COMBINE: learned gate between local and global
            combined = torch.cat([x_local, x_global], dim=-1)
            x = F.relu(self.combine_gate(combined))

        return x

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 6: Dynamic Rewiring GNN
# ─────────────────────────────────────────────────────────────────────────────

class DynamicRewireGNN(nn.Module):
    """
    Variant 6: Dynamic Rewiring GNN

    Architecture:
        - Starts from fixed structural graph (anatomical connectivity)
        - At each layer, learns a soft adjacency update based on node features
          (functional connectivity — emerges dynamically from input)
        - Functional edges = sigmoid(MLP(h_u || h_v)) — learned per forward pass
        - Final message passing combines structural + functional edges
        - Structural edges are permanent; functional edges reconfigure per input

    Inspiration: structural vs functional connectivity (2.3.5)
                 structural connectivity constrains but doesn't fully determine
                 functional connectivity; functional links emerge beyond anatomy.

    Args:
        rewire_k: number of candidate functional edges per node to consider
        rewire_threshold: minimum functional edge strength to keep (0-1)
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, rewire_k=10, rewire_threshold=0.5,
                 dropout=0.5, task="node", **kwargs):
        super().__init__()
        self.task              = task
        self.dropout           = dropout
        self.rewire_k          = rewire_k
        self.rewire_threshold  = rewire_threshold

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Structural MP layers (fixed graph)
        self.struct_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="sum")
            for _ in range(num_layers)
        ])

        # Functional edge scorer: given two node embeddings, score their
        # functional connection strength
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # functional edge strength in (0,1)
        )

        # Functional MP layers (dynamic graph)
        self.func_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(num_layers)
        ])

        # Integration: combine structural and functional representations
        self.integrate = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _functional_edges(self, x, num_nodes):
        """
        Dynamically compute functional edges based on current node embeddings.
        For each node, find top-k functionally connected nodes.
        Returns edge_index of functional connections above threshold.
        """
        k = min(self.rewire_k, num_nodes - 1)

        # Sample candidate pairs efficiently
        num_cand = min(num_nodes * k * 3, 3000)
        src = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        dst = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        mask = src != dst
        src, dst = src[mask], dst[mask]

        if src.size(0) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=x.device)

        # Score functional connection strength
        pair_feat = torch.cat([x[src], x[dst]], dim=-1)
        strength  = self.edge_scorer(pair_feat).squeeze(-1)  # [E_cand]

        # Keep edges above threshold (functional connectivity threshold)
        keep = strength > self.rewire_threshold
        if keep.sum() == 0:
            # Lower threshold if nothing passes
            keep = strength > strength.median()

        func_edge_index = torch.stack([src[keep], dst[keep]], dim=0)
        return func_edge_index

    def encode(self, x, edge_index):
        num_nodes = x.size(0)
        x = F.relu(self.input_proj(x))

        for struct_layer, func_layer, integrate in zip(
                self.struct_layers, self.func_layers, self.integrate):

            # STRUCTURAL stream: fixed anatomical graph
            x_struct = F.dropout(x, p=self.dropout, training=self.training)
            x_struct = struct_layer(x_struct, edge_index)

            # FUNCTIONAL stream: dynamically rewired graph
            func_edges = self._functional_edges(x, num_nodes)
            x_func = F.dropout(x, p=self.dropout, training=self.training)
            if func_edges.size(1) > 0:
                x_func = func_layer(x_func, func_edges)
            else:
                x_func = x_struct

            # INTEGRATE: combine structural and functional representations
            combined = torch.cat([x_struct, x_func], dim=-1)
            x = F.relu(integrate(combined))

        return x

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Update registry to include V4, V5, V6
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_CONFIGS.update({
    "V4": dict(num_shortcuts=8),
    "V5": dict(num_shortcuts=8),
    "V6": dict(rewire_k=10, rewire_threshold=0.5),
})

VARIANT_DESCRIPTIONS.update({
    "V4": "Hub-Aware GNN          — degree-scaled messages, preferential attachment",
    "V5": "Small-World Shortcut   — local MP + learned sparse long-range shortcuts",
    "V6": "Dynamic Rewiring GNN   — structural + dynamic functional edges per layer",
})

VARIANT_CLASSES.update({
    "V4": HubAwareGNN,
    "V5": SmallWorldGNN,
    "V6": DynamicRewireGNN,
})


# ─────────────────────────────────────────────────────────────────────────────
# Variant 7: Unified Brain-Inspired GNN
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedBrainGNN(nn.Module):
    """
    Variant 7: Unified Brain-Inspired GNN

    Combines all 6 biologically-inspired principles into one architecture:

        1. Modularity (V1)        — split hidden dim into independent modules
        2. Hierarchy (V2)         — low-level local + high-level global layers
        3. Hub-awareness (V4)     — degree-scaled messages
        4. Small-world (V5)       — learned sparse long-range shortcuts
        5. Dynamic rewiring (V6)  — functional edges per layer
        6. Parallel channels (V3) — multiple aggregation streams fused by attention

    Architecture per layer:
        - Split input into K modules (modularity)
        - Each module runs hub-aware message passing on structural edges (hub)
        - Each module also gets dynamic functional edges (rewiring)
        - Shortcut edges add long-range context (small-world)
        - Low layers feed into high layers via skip connections (hierarchy)
        - Final fusion via attention across all representations (parallel)

    Inspiration: all principles from sections 2.3 and 2.4
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=3, num_modules=4,
                 num_shortcuts=8, rewire_k=8,
                 dropout=0.5, task="node", **kwargs):
        super().__init__()
        assert hidden_dim % num_modules == 0, \
            "hidden_dim must be divisible by num_modules"

        self.task        = task
        self.dropout     = dropout
        self.num_modules = num_modules
        self.module_dim  = hidden_dim // num_modules
        self.hidden_dim  = hidden_dim
        self.num_shortcuts = num_shortcuts
        self.rewire_k    = rewire_k

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Module layers (standard MP — hub scaling applied manually)
        self.hub_layers = nn.ModuleList([
            nn.ModuleList([
                BasicMPLayer(self.module_dim, self.module_dim, aggr="sum")
                for _ in range(num_modules)
            ])
            for _ in range(num_layers)
        ])
        # Learned hub gate: log(degree) -> scale factor
        self.hub_gate = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, self.module_dim), nn.Sigmoid()
        )

        # Functional edge scorer (shared across layers)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Functional MP layer (dynamic rewiring stream)
        self.func_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(num_layers)
        ])

        # Shortcut attention scorer
        self.shortcut_attn = nn.Linear(hidden_dim * 2, 1)

        # Shortcut MP layer (small-world stream)
        self.shortcut_layers = nn.ModuleList([
            BasicMPLayer(hidden_dim, hidden_dim, aggr="mean")
            for _ in range(num_layers)
        ])

        # Sparse inter-module gates (modularity)
        self.inter_module_gates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])
        sparsity = 0.3
        for i in range(num_layers):
            mask = torch.zeros(hidden_dim, hidden_dim)
            for a in range(num_modules):
                for b in range(num_modules):
                    if a != b:
                        sa, sb = a * self.module_dim, b * self.module_dim
                        submask = (torch.rand(self.module_dim, self.module_dim) < sparsity).float()
                        mask[sa:sa+self.module_dim, sb:sb+self.module_dim] = submask
            self.register_buffer(f"sparse_mask_{i}", mask)

        # Hierarchical skip projection
        # After num_layers we have: input + num_layers layer outputs
        total_skips = 1 + num_layers
        self.hier_fusion = nn.Linear(hidden_dim * total_skips, hidden_dim)

        # Final attention fusion across 3 streams (modular, functional, shortcut)
        self.stream_attn = nn.Linear(hidden_dim, 3)
        self.stream_proj = nn.Linear(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _functional_edges(self, x, num_nodes):
        k = min(self.rewire_k, num_nodes - 1)
        num_cand = min(num_nodes * k * 3, 3000)
        src = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        dst = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        if src.size(0) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=x.device)
        pair_feat = torch.cat([x[src], x[dst]], dim=-1)
        strength  = self.edge_scorer(pair_feat).squeeze(-1)
        keep = strength > 0.5
        if keep.sum() == 0:
            keep = strength > strength.median()
        return torch.stack([src[keep], dst[keep]], dim=0)

    def _shortcut_edges(self, x, num_nodes):
        k = min(self.num_shortcuts, num_nodes - 1)
        num_cand = min(num_nodes * k * 2, 2000)
        src = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        dst = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        if src.size(0) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=x.device)
        pair_feat   = torch.cat([x[src], x[dst]], dim=-1)
        scores      = self.shortcut_attn(pair_feat).squeeze(-1)
        num_keep    = min(k * num_nodes, src.size(0))
        _, top_idx  = scores.topk(num_keep)
        return torch.stack([src[top_idx], dst[top_idx]], dim=0)

    def encode(self, x, edge_index):
        from torch_geometric.utils import degree
        num_nodes = x.size(0)
        x = F.relu(self.input_proj(x))
        deg = degree(edge_index[0], num_nodes=num_nodes)

        all_skips = [x]  # for hierarchical fusion

        for layer_idx in range(len(self.hub_layers)):
            hub_blocks   = self.hub_layers[layer_idx]
            func_layer   = self.func_layers[layer_idx]
            short_layer  = self.shortcut_layers[layer_idx]
            gate         = self.inter_module_gates[layer_idx]
            sparse_mask  = getattr(self, f"sparse_mask_{layer_idx}")

            # ── Stream 1: Modular + Hub-aware ──────────────────────────
            module_outs = []
            # Hub scaling: scale each node embedding by its degree importance
            log_deg = torch.log1p(deg.float()).unsqueeze(-1)  # [N, 1]
            hub_scale = self.hub_gate(log_deg)                # [N, module_dim]

            for k, hub_block in enumerate(hub_blocks):
                s = k * self.module_dim
                e = s + self.module_dim
                x_k = x[:, s:e] * hub_scale   # scale input by hub importance
                x_k = F.dropout(x_k, p=self.dropout, training=self.training)
                h_k = hub_block(x_k, edge_index)
                module_outs.append(h_k)
            x_modular = torch.cat(module_outs, dim=-1)  # [N, hidden]

            # Sparse inter-module communication
            sparse_w    = gate.weight * sparse_mask
            inter       = torch.sigmoid(F.linear(x_modular, sparse_w))
            x_modular   = x_modular + 0.1 * inter

            # ── Stream 2: Dynamic functional rewiring ──────────────────
            func_edges  = self._functional_edges(x, num_nodes)
            x_func      = F.dropout(x, p=self.dropout, training=self.training)
            if func_edges.size(1) > 0:
                x_func  = func_layer(x_func, func_edges)
            else:
                x_func  = x_modular

            # ── Stream 3: Small-world shortcuts ────────────────────────
            short_edges = self._shortcut_edges(x, num_nodes)
            x_short     = F.dropout(x, p=self.dropout, training=self.training)
            if short_edges.size(1) > 0:
                x_short = short_layer(x_short, short_edges)
            else:
                x_short = x_modular

            # ── Parallel attention fusion across 3 streams ─────────────
            # Attention weights from mean of all streams
            mean_rep    = (x_modular + x_func + x_short) / 3.0
            attn_w      = F.softmax(self.stream_attn(mean_rep), dim=-1)  # [N, 3]
            x_fused     = (x_modular * attn_w[:, 0:1] +
                           x_func    * attn_w[:, 1:2] +
                           x_short   * attn_w[:, 2:3])
            x = F.relu(self.stream_proj(x_fused))
            x = F.dropout(x, p=self.dropout, training=self.training)

            all_skips.append(x)  # hierarchical skip

        # ── Hierarchical fusion: combine all layer outputs ──────────────
        x = F.relu(self.hier_fusion(torch.cat(all_skips, dim=-1)))
        return x

    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.classifier(h)


# Update registry
VARIANT_CONFIGS.update({
    "V7": dict(num_modules=4, num_shortcuts=8, rewire_k=8),
})

VARIANT_DESCRIPTIONS.update({
    "V7": "Unified Brain-Inspired  — modularity+hierarchy+hubs+shortcuts+rewiring",
})

VARIANT_CLASSES.update({
    "V7": UnifiedBrainGNN,
})