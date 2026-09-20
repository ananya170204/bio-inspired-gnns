# Biologically Inspired Graph Neural Networks

This repository provides code and data for the paper **"Learning from Biological Networks: Which Organisational Principles Improve Graph Neural Networks?"** 
It implements a controlled framework for evaluating six biological and network-organisational principles as architectural inductive biases within Message Passing Neural Networks (MPNNs). Eight baseline configurations (C1–C8) and seven principle-specific variants (V1–V7) are evaluated across seven datasets spanning citation, molecular, biological, and social-network domains.

## Architectures

**Baseline Configurations (C1–C8):** Eight controlled MPNNs varying aggregation function, update mechanism, and activation function:
* C1: Sum + Linear + ReLU
* C2: Mean + Linear + ReLU
* C3: Max + Linear + ReLU
* C4: Sum + MLP + ReLU
* C5: Mean + MLP + ReLU
* C6: Sum + MLP + ELU
* C7: Sum + MLP + Tanh
* C8: Sum + Mean + Max + MLP + ReLU (multi-aggregation)

**Principle-Specific Variants (V1–V7):** Seven architectures operationalising biological and network-organisational principles:
* V1: Modularity and sparsity
* V2: Hierarchical multi-scale integration
* V3: Parallel processing
* V4: Hub-aware processing (preferential attachment)
* V5: Small-world connectivity
* V6: Structural–functional connectivity
* V7: Unified architecture (all six principles combined)

## Datasets

Seven datasets are used across node- and graph-classification tasks:
* Cora: node classification, 7 classes, 2,708 nodes
* CiteSeer: node classification, 6 classes, 3,327 nodes
* Facebook (SNAP): node classification, 15 Louvain communities, 4,039 nodes
* Twitter (SNAP): node classification, 59 Louvain communities, 10,000 nodes
* MUTAG: graph classification, 2 classes, 188 graphs
* PROTEINS: graph classification, 2 classes, 1,113 graphs
* ENZYMES: graph classification, 6 classes, 600 graphs

Facebook and Twitter use topology-derived node features (degree, clustering coefficient, PageRank, and triangle count) with community labels generated via the Louvain algorithm.

## Experimental Protocol

* 10 independent runs per model with different random seeds
* 200 training epochs per run; best validation checkpoint selected for testing
* Performance reported as mean ± standard deviation across runs
* Evaluation metrics: accuracy, macro-F1, precision, and recall
* Statistical analysis: Friedman test and Nemenyi post-hoc test across 6 datasets (Twitter excluded from formal analysis due to extreme class imbalance)

## Folder and File Descriptor

* `gnn_baseline/models/mpnn_base.py`: baseline MPNN configurations (C1–C8) and factory function
* `gnn_baseline/datasets/loader.py`: dataset loading and preprocessing
* `gnn_baseline/train.py`: trainer and benchmark runner
* `gnn_baseline/main.py`: entry point for baseline experiments
* `gnn_baseline/run_variants.py`: entry point for variant experiments
* `run_all.sh`: shell script to run all experiments sequentially
* `requirements.txt`: Python dependencies

## Usage

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
bash run_all.sh
```

Results are saved to the results/ folder, with three output files per dataset: for example, cora_results.json (baseline performance across 10 runs), cora_table.csv (baseline summary table), and cora_variants.json (variant performance across 10 runs)


```

## License

MIT License