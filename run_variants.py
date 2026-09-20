"""
Run proposed variants and compare against baseline configs.
Edit the parameters below to change dataset/epochs/etc.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, time, statistics, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from models.variants import get_variant, VARIANT_CONFIGS, VARIANT_DESCRIPTIONS
from models.mpnn_base import get_model, CONFIGS, CONFIG_DESCRIPTIONS
from datasets.loader import load_dataset, get_graph_loaders

# ── Change these to run different experiments ──
DATASET    = "cora"
EPOCHS     = 200
RUNS       = 10
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.1
BATCH_SIZE = 32
SEED       = 42
# ──────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(preds, labels):
    p = preds.numpy() if hasattr(preds, 'numpy') else preds
    l = labels.numpy() if hasattr(labels, 'numpy') else labels
    return {
        "accuracy":  accuracy_score(l, p),
        "f1":        f1_score(l, p, average="macro", zero_division=0),
        "precision": precision_score(l, p, average="macro", zero_division=0),
        "recall":    recall_score(l, p, average="macro", zero_division=0),
    }


def train_eval(model, task, data_obj, epochs, dropout, batch_size):
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    model       = model.to(device)
    optimizer   = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler   = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    best_val    = 0.0
    best_test   = 0.0
    best_preds  = None
    best_labels = None

    if task == "node_classification":
        data = data_obj.to(device)
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out  = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out    = model(data.x, data.edge_index)
                val_a  = (out[data.val_mask].argmax(-1) == data.y[data.val_mask]).float().mean().item()
                preds  = out[data.test_mask].argmax(-1).cpu()
                labels = data.y[data.test_mask].cpu()
                test_a = (preds == labels).float().mean().item()
            scheduler.step(val_a)
            if val_a > best_val:
                best_val    = val_a
                best_test   = test_a
                best_preds  = preds
                best_labels = labels

    else:
        tr, val, te = get_graph_loaders(data_obj, batch_size)

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in tr:
                batch = batch.to(device)
                optimizer.zero_grad()
                out  = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                # Validation accuracy
                vc = vt = 0
                for b in val:
                    b = b.to(device)
                    p = model(b.x, b.edge_index, b.batch).argmax(-1)
                    vc += (p == b.y).sum().item()
                    vt += b.num_graphs
                val_a = vc / vt if vt else 0

                # Test predictions
                all_preds, all_labels = [], []
                for b in te:
                    b = b.to(device)
                    p = model(b.x, b.edge_index, b.batch).argmax(-1).cpu()
                    all_preds.append(p)
                    all_labels.append(b.y.cpu())
                preds  = torch.cat(all_preds)
                labels = torch.cat(all_labels)
                test_a = (preds == labels).float().mean().item()

            scheduler.step(val_a)
            if val_a > best_val:
                best_val    = val_a
                best_test   = test_a
                best_preds  = preds
                best_labels = labels

    return best_val, best_test, best_preds, best_labels


def main():
    set_seed(SEED)

    data_obj, task, nf, nc = load_dataset(DATASET)
    task_short = task.split("_")[0]
    results = {}

    print(f"\n{'='*62}")
    print(f" Dataset: {DATASET.upper()}  |  Features: {nf}  |  Classes: {nc}")
    print(f"{'='*62}")

    print(f"\n── Proposed variants ──")
    for v_name in VARIANT_CONFIGS:
        print(f"  [{v_name}] {VARIANT_DESCRIPTIONS[v_name]}")
        vals, tests, times = [], [], []
        all_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

        for r in range(RUNS):
            set_seed(SEED + r)
            model = get_variant(v_name, nf, HIDDEN_DIM, nc,
                                num_layers=NUM_LAYERS,
                                dropout=DROPOUT, task=task_short)
            t0 = time.time()
            v, t, preds, labels = train_eval(model, task, data_obj,
                                             EPOCHS, DROPOUT, BATCH_SIZE)
            vals.append(v)
            tests.append(t)
            times.append(time.time() - t0)

            m = compute_metrics(preds, labels)
            for k in all_metrics:
                all_metrics[k].append(m[k])

        results[v_name] = {
            "type":           "variant",
            "description":    VARIANT_DESCRIPTIONS[v_name],
            "val_mean":       statistics.mean(vals),
            "val_std":        statistics.stdev(vals) if RUNS > 1 else 0,
            "test_mean":      statistics.mean(tests),
            "test_std":       statistics.stdev(tests) if RUNS > 1 else 0,
            "accuracy_mean":  statistics.mean(all_metrics["accuracy"]),
            "accuracy_std":   statistics.stdev(all_metrics["accuracy"]) if RUNS > 1 else 0,
            "f1_mean":        statistics.mean(all_metrics["f1"]),
            "f1_std":         statistics.stdev(all_metrics["f1"]) if RUNS > 1 else 0,
            "precision_mean": statistics.mean(all_metrics["precision"]),
            "precision_std":  statistics.stdev(all_metrics["precision"]) if RUNS > 1 else 0,
            "recall_mean":    statistics.mean(all_metrics["recall"]),
            "recall_std":     statistics.stdev(all_metrics["recall"]) if RUNS > 1 else 0,
            "avg_time":       statistics.mean(times),
            "num_params":     sum(p.numel() for p in model.parameters()
                                  if p.requires_grad),
            "runs_accuracy":  all_metrics["accuracy"],
            "runs_f1":        all_metrics["f1"],
            "runs_precision": all_metrics["precision"],
            "runs_recall":    all_metrics["recall"],
        }
        r = results[v_name]
        print(f"        → test {r['test_mean']:.4f} ± {r['test_std']:.4f}"
              f"  |  F1 {r['f1_mean']:.4f}"
              f"  P {r['precision_mean']:.4f}"
              f"  R {r['recall_mean']:.4f}")

    # ── Summary table ──────────────────────────────────────────────
    best = max(r["test_mean"] for r in results.values())
    print(f"\n{'='*72}")
    print(f" Summary — {DATASET.upper()}")
    print(f"{'='*72}")
    print(f"{'Name':<6} {'Type':<9} {'Description':<38} {'Test':>12} {'Params':>9}")
    print(f"{'-'*72}")
    for name, m in sorted(results.items(), key=lambda x: -x[1]["test_mean"]):
        marker = " ◄" if abs(m["test_mean"] - best) < 1e-6 else ""
        print(f"{name:<6} {m['type']:<9} {m['description'][:38]:<38} "
              f"{m['test_mean']:>6.4f}±{m['test_std']:.3f} "
              f"{m['num_params']:>9,}{marker}")
    print("=" * 72)

    Path("results").mkdir(exist_ok=True)
    out = f"results/{DATASET}_variants.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()