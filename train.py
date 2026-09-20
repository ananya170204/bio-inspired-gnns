"""
Training & Evaluation Engine
"""
import random
import numpy as np

import time, csv, json, statistics
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, accuracy_score)

from models.mpnn_base import get_model, CONFIGS, CONFIG_DESCRIPTIONS
from datasets.loader import load_dataset, get_graph_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, lr=1e-3, weight_decay=5e-4, device=None):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max",
                                           factor=0.5, patience=10)

    # ── Node classification ──────────────────────────────────────────────────

    def train_node(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)
        out  = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    @torch.no_grad()
    def eval_node(self, data, mask_name="test_mask"):
        """Returns (accuracy, preds, labels)."""
        self.model.eval()
        data  = data.to(self.device)
        out   = self.model(data.x, data.edge_index)
        mask  = getattr(data, mask_name)
        pred  = out[mask].argmax(dim=-1).cpu()
        label = data.y[mask].cpu()
        acc   = (pred == label).float().mean().item()
        return acc, pred, label

    # ── Graph classification ─────────────────────────────────────────────────

    def train_graph(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out  = self.model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().item() * batch.num_graphs
        return total_loss / sum(b.num_graphs for b in loader)

    @torch.no_grad()
    def eval_graph(self, loader):
        """Returns (accuracy, preds, labels)."""
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            pred  = self.model(batch.x, batch.edge_index,
                               batch.batch).argmax(-1).cpu()
            all_preds.append(pred)
            all_labels.append(batch.y.cpu())
        if not all_preds:
            return 0.0, torch.tensor([]), torch.tensor([])
        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc    = (preds == labels).float().mean().item()
        return acc, preds, labels

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, task, epochs=200, verbose=True, **loaders):
        best_val  = 0.0
        best_test = 0.0
        best_preds  = None
        best_labels = None

        for epoch in range(1, epochs + 1):
            if task == "node_classification":
                loss            = self.train_node(loaders["data"])
                val_a, _, _     = self.eval_node(loaders["data"], "val_mask")
                test_a, tp, tl  = self.eval_node(loaders["data"], "test_mask")
            else:
                loss            = self.train_graph(loaders["train_loader"])
                val_a, _, _     = self.eval_graph(loaders["val_loader"])
                test_a, tp, tl  = self.eval_graph(loaders["test_loader"])

            self.scheduler.step(val_a)
            if val_a > best_val:
                best_val    = val_a
                best_test   = test_a
                best_preds  = tp
                best_labels = tl

            if verbose and epoch % 50 == 0:
                print(f"  ep {epoch:>3d}  loss={loss:.4f}  "
                      f"val={val_a:.4f}  test={test_a:.4f}")

        return {
            "best_val":   best_val,
            "best_test":  best_test,
            "preds":      best_preds,
            "labels":     best_labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels):
    """Compute accuracy, macro F1, macro precision, macro recall."""
    p = preds.numpy()
    l = labels.numpy()
    return {
        "accuracy":  accuracy_score(l, p),
        "f1":        f1_score(l, p, average="macro", zero_division=0),
        "precision": precision_score(l, p, average="macro", zero_division=0),
        "recall":    recall_score(l, p, average="macro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner — baselines
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_all(dataset_name, hidden_dim=64, num_layers=3, epochs=200,
                  runs=3, dropout=0.5, batch_size=32,
                  out_dir="./results", configs=None):

    configs = configs or list(CONFIGS.keys())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*58}")
    print(f" Dataset: {dataset_name.upper()}")
    print(f"{'='*58}")

    data_obj, task, num_features, num_classes = load_dataset(dataset_name)
    print(f" Task: {task}  |  Features: {num_features}  |  Classes: {num_classes}")

    all_results = {}

    for cfg_name in configs:
        desc = CONFIG_DESCRIPTIONS[cfg_name]
        print(f"\n  [{cfg_name}] {desc}")

        run_vals     = []
        run_tests    = []
        run_times    = []
        run_metrics  = {"accuracy": [], "f1": [], "precision": [], "recall": []}

        for run in range(runs):
            random.seed(42 + run)
            np.random.seed(42 + run)
            torch.manual_seed(42 + run)
            model   = get_model(cfg_name, num_features, hidden_dim,
                                num_classes, num_layers=num_layers,
                                dropout=dropout, task=task.split("_")[0])
            trainer = Trainer(model, lr=1e-3)
            t0      = time.time()

            if task == "node_classification":
                res = trainer.fit(task, epochs=epochs, verbose=False,
                                  data=data_obj)
            else:
                tr, val, te = get_graph_loaders(data_obj, batch_size,
                                                seed=run * 7)
                res = trainer.fit(task, epochs=epochs, verbose=False,
                                  train_loader=tr, val_loader=val,
                                  test_loader=te)

            run_vals.append(res["best_val"])
            run_tests.append(res["best_test"])
            run_times.append(time.time() - t0)

            m = compute_metrics(res["preds"], res["labels"])
            for k in run_metrics:
                run_metrics[k].append(m[k])
        entry = {
            "description":      desc,
            "val_mean":         statistics.mean(run_vals),
            "val_std":          statistics.stdev(run_vals) if runs > 1 else 0,
            "test_mean":        statistics.mean(run_tests),
            "test_std":         statistics.stdev(run_tests) if runs > 1 else 0,
            "accuracy_mean":    statistics.mean(run_metrics["accuracy"]),
            "accuracy_std":     statistics.stdev(run_metrics["accuracy"]) if runs > 1 else 0,
            "f1_mean":          statistics.mean(run_metrics["f1"]),
            "f1_std":           statistics.stdev(run_metrics["f1"]) if runs > 1 else 0,
            "precision_mean":   statistics.mean(run_metrics["precision"]),
            "precision_std":    statistics.stdev(run_metrics["precision"]) if runs > 1 else 0,
            "recall_mean":      statistics.mean(run_metrics["recall"]),
            "recall_std":       statistics.stdev(run_metrics["recall"]) if runs > 1 else 0,
            "avg_time_s":       statistics.mean(run_times),
            "num_params":       sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
            # ADD THESE
            "runs_accuracy":    run_metrics["accuracy"],
            "runs_f1":          run_metrics["f1"],
            "runs_precision":   run_metrics["precision"],
            "runs_recall":      run_metrics["recall"],
        }
        all_results[cfg_name] = entry
        print(f"        → test {entry['test_mean']:.4f} ± {entry['test_std']:.4f}"
              f"  |  F1 {entry['f1_mean']:.4f}  "
              f"P {entry['precision_mean']:.4f}  "
              f"R {entry['recall_mean']:.4f}")

    out_file = Path(out_dir) / f"{dataset_name}_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner — variants
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_variants(dataset_name, hidden_dim=64, num_layers=3, epochs=200,
                       runs=3, dropout=0.5, batch_size=32,
                       out_dir="./results", variants=None):
    from models.variants import get_variant, VARIANT_CONFIGS, VARIANT_DESCRIPTIONS

    variants = variants or list(VARIANT_CONFIGS.keys())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f" Dataset: {dataset_name.upper()}  |  Variants")
    print(f"{'='*62}")

    data_obj, task, num_features, num_classes = load_dataset(dataset_name)
    print(f" Task: {task}  |  Features: {num_features}  |  Classes: {num_classes}")

    all_results = {}

    for v_name in variants:
        desc = VARIANT_DESCRIPTIONS[v_name]
        print(f"\n  [{v_name}] {desc}")

        run_vals    = []
        run_tests   = []
        run_times   = []
        run_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

        for run in range(runs):
            model = get_variant(v_name, num_features, hidden_dim,
                                num_classes, num_layers=num_layers,
                                dropout=dropout,
                                task=task.split("_")[0])
            trainer = Trainer(model, lr=1e-3)
            t0 = time.time()

            if task == "node_classification":
                res = trainer.fit(task, epochs=epochs, verbose=False,
                                  data=data_obj)
            else:
                tr, val, te = get_graph_loaders(data_obj, batch_size,
                                                seed=run * 7)
                res = trainer.fit(task, epochs=epochs, verbose=False,
                                  train_loader=tr, val_loader=val,
                                  test_loader=te)

            run_vals.append(res["best_val"])
            run_tests.append(res["best_test"])
            run_times.append(time.time() - t0)

            m = compute_metrics(res["preds"], res["labels"])
            for k in run_metrics:
                run_metrics[k].append(m[k])

        entry = {
            "type":           "variant",
            "description":    desc,
            "val_mean":       statistics.mean(run_vals),
            "val_std":        statistics.stdev(run_vals) if runs > 1 else 0,
            "test_mean":      statistics.mean(run_tests),
            "test_std":       statistics.stdev(run_tests) if runs > 1 else 0,
            "accuracy_mean":  statistics.mean(run_metrics["accuracy"]),
            "accuracy_std":   statistics.stdev(run_metrics["accuracy"]) if runs > 1 else 0,
            "f1_mean":        statistics.mean(run_metrics["f1"]),
            "f1_std":         statistics.stdev(run_metrics["f1"]) if runs > 1 else 0,
            "precision_mean": statistics.mean(run_metrics["precision"]),
            "precision_std":  statistics.stdev(run_metrics["precision"]) if runs > 1 else 0,
            "recall_mean":    statistics.mean(run_metrics["recall"]),
            "recall_std":     statistics.stdev(run_metrics["recall"]) if runs > 1 else 0,
            "avg_time":       statistics.mean(run_times),
            "num_params":     sum(p.numel() for p in model.parameters()
                                  if p.requires_grad),
        }
        all_results[v_name] = entry
        print(f"        → test {entry['test_mean']:.4f} ± {entry['test_std']:.4f}"
              f"  |  F1 {entry['f1_mean']:.4f}  "
              f"P {entry['precision_mean']:.4f}  "
              f"R {entry['recall_mean']:.4f}")

    out_file = Path(out_dir) / f"{dataset_name}_variants.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty table + CSV export
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: Dict, dataset_name: str = ""):
    print(f"\n{'='*72}")
    print(f" Results — {dataset_name}")
    print(f"{'='*72}")
    print(f"{'Config':<6} {'Aggregation':<8} {'Update':<8} {'Act':<6} "
          f"{'Val':>8} {'Test':>12} {'Params':>9}")
    print(f"{'-'*72}")

    best = max(v["test_mean"] for v in results.values())
    for name, m in sorted(results.items(), key=lambda x: -x[1]["test_mean"]):
        from models.mpnn_base import CONFIGS
        c      = CONFIGS[name]
        marker = " ◄" if abs(m["test_mean"] - best) < 1e-6 else ""
        print(f"{name:<6} {c['aggr']:<8} {c['update']:<8} {c['activation']:<6} "
              f"{m['val_mean']:>6.4f}±{m['val_std']:.3f} "
              f"{m['test_mean']:>8.4f}±{m['test_std']:.3f} "
              f"{m['num_params']:>9,}{marker}")
    print("=" * 72)


def export_csv(results, dataset_name, out_dir="./results"):
    Path(out_dir).mkdir(exist_ok=True)
    path = Path(out_dir) / f"{dataset_name}_table.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "aggr", "update", "activation",
                    "val_mean", "val_std", "test_mean", "test_std",
                    "accuracy_mean", "accuracy_std",
                    "f1_mean", "f1_std",
                    "precision_mean", "precision_std",
                    "recall_mean", "recall_std",
                    "num_params", "avg_time_s"])
        from models.mpnn_base import CONFIGS
        for name, m in results.items():
            c = CONFIGS.get(name, {"aggr": "—", "update": "—", "activation": "—"})
            w.writerow([
                name, c.get("aggr","—"), c.get("update","—"),
                c.get("activation","—"),
                m["val_mean"], m["val_std"],
                m["test_mean"], m["test_std"],
                m.get("accuracy_mean", m["test_mean"]),
                m.get("accuracy_std",  m["test_std"]),
                m.get("f1_mean", ""),        m.get("f1_std", ""),
                m.get("precision_mean", ""), m.get("precision_std", ""),
                m.get("recall_mean", ""),    m.get("recall_std", ""),
                m["num_params"],
                m.get("avg_time_s", m.get("avg_time", 0)),
            ])
    print(f"  CSV → {path}")