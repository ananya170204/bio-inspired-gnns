import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, torch, random
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train import benchmark_all, print_results_table, export_csv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ── Change these to run different experiments ──
DATASET    = "cora"
EPOCHS     = 200
RUNS       = 10
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.1
BATCH_SIZE = 32
OUT_DIR    = "./results"
CONFIGS    = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]   # None = all 8, or e.g. ["C1", "C4", "C8"]
# ──────────────────────────────────────────────

results = benchmark_all(
    dataset_name = DATASET,
    hidden_dim   = HIDDEN_DIM,
    num_layers   = NUM_LAYERS,
    epochs       = EPOCHS,
    runs         = RUNS,
    dropout      = DROPOUT,
    batch_size   = BATCH_SIZE,
    out_dir      = OUT_DIR,
    configs      = CONFIGS,
)
print_results_table(results, DATASET.upper())
export_csv(results, DATASET, OUT_DIR)