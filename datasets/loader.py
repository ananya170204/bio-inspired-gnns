"""
Dataset Loading Module
======================
Handles 7 benchmark datasets:
    1. Facebook Ego-Networks  (SNAP)      - node classification (Louvain communities)
    2. Twitter Ego-Networks   (SNAP)      - node classification (Louvain communities)
    3. Cora                   (Planetoid) - node classification
    4. CiteSeer               (Planetoid) - node classification
    5. MUTAG                  (TUDataset) - graph classification
    6. PROTEINS               (TUDataset) - graph classification
    7. ENZYMES                (TUDataset) - graph classification

Facebook and Twitter use Louvain community detection for labels
and structural features (degree, clustering, pagerank, triangles).
These are meaningful social network tasks based on real graph structure.
"""

import os
import torch
import numpy as np
import urllib.request
from pathlib import Path

from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader

DATA_ROOT = Path("./data")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_degree_histogram(dataset):
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


# ─────────────────────────────────────────────────────────────────────────────
# Shared social network loader (Facebook and Twitter)
# ─────────────────────────────────────────────────────────────────────────────

def _structural_features(G, num_nodes):
    """Engineer 4 structural features from graph G."""
    import networkx as nx
    deg        = np.array([d for _, d in G.degree()], dtype=np.float32)
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)
    try:
        pagerank = np.array(list(nx.pagerank(G, max_iter=100).values()),
                            dtype=np.float32)
    except Exception:
        pagerank = np.zeros(num_nodes, dtype=np.float32)
    triangles  = np.array([nx.triangles(G, n) for n in G.nodes()],
                           dtype=np.float32)
    x = np.stack([deg, clustering, pagerank, triangles], axis=1)
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)
    return torch.tensor(x, dtype=torch.float)


def _louvain_labels(G):
    """
    Apply Louvain community detection to get meaningful node labels.
    Returns labels based on actual graph community structure — not degree buckets.
    Communities are found by maximising modularity, directly connecting to
    the modularity principle in Section 2.4.3.
    """
    import community as community_louvain
    partition   = community_louvain.best_partition(G, random_state=42)
    raw_labels  = np.array([partition[i] for i in range(len(G.nodes()))],
                            dtype=np.int64)
    # Remap to contiguous 0..K-1
    unique      = np.unique(raw_labels)
    label_map   = {old: new for new, old in enumerate(unique)}
    labels      = np.array([label_map[l] for l in raw_labels], dtype=np.int64)
    num_classes = len(unique)
    return torch.tensor(labels, dtype=torch.long), num_classes


def _build_social_graph(edge_file, max_nodes=None):
    """Load edge list and build undirected networkx graph."""
    import networkx as nx
    G = nx.read_edgelist(edge_file, nodetype=int, create_using=nx.Graph())
    G = nx.convert_node_labels_to_integers(G)
    if max_nodes is not None:
        # Take largest connected component capped at max_nodes
        gcc   = max(nx.connected_components(G), key=len)
        nodes = sorted(gcc)[:max_nodes]
        G     = G.subgraph(nodes).copy()
        G     = nx.convert_node_labels_to_integers(G)
    return G


def _to_pyg_data(G):
    """Convert networkx graph to PyG Data with structural features + Louvain labels."""
    num_nodes  = G.number_of_nodes()
    edges      = list(G.edges())
    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]),
                               torch.cat([dst, src])], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    x              = _structural_features(G, num_nodes)
    y, num_classes = _louvain_labels(G)

    return Data(x=x, edge_index=edge_index, y=y,
                num_nodes=num_nodes), num_classes


def _add_masks(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Add train/val/test masks to a node classification Data object."""
    n    = data.num_nodes
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    t1, t2 = int(train_ratio * n), int((train_ratio + val_ratio) * n)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask   = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)
    data.train_mask[perm[:t1]]  = True
    data.val_mask[perm[t1:t2]]  = True
    data.test_mask[perm[t2:]]   = True
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: Facebook Ego-Network
# ─────────────────────────────────────────────────────────────────────────────

def load_facebook(root=None):
    """
    Facebook ego-network with Louvain community labels.

    Features: 4 structural (degree, clustering, pagerank, triangles)
    Labels:   Louvain community membership — real graph communities
              found by maximising modularity, directly motivated by
              Section 2.4.3 (Modularity and Community Structure)

    This is a meaningful task: predict which social community a person
    belongs to based on their network position and local structure.
    """
    import networkx as nx
    root      = root or str(DATA_ROOT / "facebook")
    os.makedirs(root, exist_ok=True)
    edge_file = os.path.join(root, "facebook_combined.txt")

    if not os.path.exists(edge_file):
        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
        gz  = edge_file + ".gz"
        print(f"[Facebook] Downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, gz)
            import gzip, shutil
            with gzip.open(gz, 'rb') as fi, open(edge_file, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(gz)
        except Exception as e:
            print(f"[Facebook] Download failed: {e}. Using synthetic graph.")
            G = nx.barabasi_albert_graph(4039, 5, seed=42)
            G = nx.convert_node_labels_to_integers(G)
            data, nc = _to_pyg_data(G)
            data = _add_masks(data)
            print(f"[Facebook-Synthetic] {data.num_nodes} nodes, {nc} communities")
            return data, 4, nc

    G    = _build_social_graph(edge_file)
    data, nc = _to_pyg_data(G)
    data = _add_masks(data)
    print(f"[Facebook] Loaded: {data.num_nodes} nodes, "
          f"{data.edge_index.size(1)//2} edges, {nc} Louvain communities")
    return data, 4, nc


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: Twitter Ego-Network
# ─────────────────────────────────────────────────────────────────────────────

def load_twitter(root=None):
    """
    Twitter ego-network with Louvain community labels.
    Subsampled to 10,000 nodes from largest connected component.

    Features: 4 structural (degree, clustering, pagerank, triangles)
    Labels:   Louvain community membership — real graph communities
    """
    import networkx as nx
    root      = root or str(DATA_ROOT / "twitter")
    os.makedirs(root, exist_ok=True)
    edge_file = os.path.join(root, "twitter_combined.txt")

    if not os.path.exists(edge_file):
        url = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
        gz  = edge_file + ".gz"
        print(f"[Twitter] Downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, gz)
            import gzip, shutil
            with gzip.open(gz, 'rb') as fi, open(edge_file, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(gz)
        except Exception as e:
            print(f"[Twitter] Download failed: {e}. Using synthetic graph.")
            G = nx.barabasi_albert_graph(10000, 5, seed=42)
            G = nx.convert_node_labels_to_integers(G)
            data, nc = _to_pyg_data(G)
            data = _add_masks(data)
            print(f"[Twitter-Synthetic] {data.num_nodes} nodes, {nc} communities")
            return data, 4, nc

    print("[Twitter] Building graph (this may take a minute)...")
    G    = _build_social_graph(edge_file, max_nodes=10000)
    data, nc = _to_pyg_data(G)
    data = _add_masks(data)
    print(f"[Twitter] Loaded: {data.num_nodes} nodes, "
          f"{data.edge_index.size(1)//2} edges, {nc} Louvain communities")
    return data, 4, nc


# ─────────────────────────────────────────────────────────────────────────────
# Datasets 3 & 4: Cora / CiteSeer
# ─────────────────────────────────────────────────────────────────────────────

def load_planetoid(name="Cora", root=None):
    root = root or str(DATA_ROOT / name.lower())
    return Planetoid(root=root, name=name, transform=NormalizeFeatures())


# ─────────────────────────────────────────────────────────────────────────────
# Datasets 5, 6, 7: MUTAG / PROTEINS / ENZYMES
# ─────────────────────────────────────────────────────────────────────────────

def load_tu_dataset(name="MUTAG", root=None):
    root = root or str(DATA_ROOT / name.lower())
    return TUDataset(root=root, name=name, use_node_attr=True)


def load_enzymes(root=None):
    root = root or str(DATA_ROOT / "enzymes")
    ds   = TUDataset(root=root, name="ENZYMES", use_node_attr=True)
    nf   = ds.num_node_features if ds.num_node_features > 0 else 1
    print(f"[ENZYMES] Loaded: {len(ds)} graphs, {nf} features, "
          f"{ds.num_classes} classes")
    return ds, nf, int(ds.num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Unified loader
# ─────────────────────────────────────────────────────────────────────────────

DATASET_INFO = {
    "facebook": {"task": "node_classification",
                 "description": "Facebook ego-network — Louvain community detection labels"},
    "twitter":  {"task": "node_classification",
                 "description": "Twitter ego-network — Louvain community detection labels"},
    "cora":     {"task": "node_classification",
                 "description": "Citation network (2708 nodes, 7 classes)"},
    "citeseer": {"task": "node_classification",
                 "description": "Citation network (3327 nodes, 6 classes)"},
    "mutag":    {"task": "graph_classification",
                 "description": "Mutagenic compounds (188 graphs, 2 classes)"},
    "proteins": {"task": "graph_classification",
                 "description": "Protein graphs (1113 graphs, 2 classes)"},
    "enzymes":  {"task": "graph_classification",
                 "description": "Enzyme biological graphs (600 graphs, 6 classes)"},
}


def load_dataset(name, root=None):
    name = name.lower()
    if name == "facebook":
        data, nf, nc = load_facebook(root=root)
        return data, "node_classification", nf, nc
    elif name == "twitter":
        data, nf, nc = load_twitter(root=root)
        return data, "node_classification", nf, nc
    elif name in ("cora", "citeseer"):
        ds = load_planetoid(name.capitalize(), root=root)
        return ds[0], "node_classification", ds.num_node_features, ds.num_classes
    elif name == "mutag":
        ds = load_tu_dataset("MUTAG", root=root)
        nf = ds.num_node_features if ds.num_node_features > 0 else 1
        return ds, "graph_classification", nf, ds.num_classes
    elif name == "proteins":
        ds = load_tu_dataset("PROTEINS", root=root)
        nf = ds.num_node_features if ds.num_node_features > 0 else 1
        return ds, "graph_classification", nf, ds.num_classes
    elif name == "enzymes":
        ds, nf, nc = load_enzymes(root=root)
        return ds, "graph_classification", nf, nc
    else:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from: {list(DATASET_INFO.keys())}")


def get_graph_loaders(dataset, batch_size=32, val_ratio=0.1,
                      test_ratio=0.1, seed=42):
    n      = len(dataset)
    rng    = torch.Generator().manual_seed(seed)
    perm   = torch.randperm(n, generator=rng).tolist()
    n_test = int(test_ratio * n)
    n_val  = int(val_ratio  * n)
    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    train_loader = DataLoader([dataset[i] for i in train_idx],
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader([dataset[i] for i in val_idx],
                              batch_size=batch_size)
    test_loader  = DataLoader([dataset[i] for i in test_idx],
                              batch_size=batch_size)
    return train_loader, val_loader, test_loader