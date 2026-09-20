#!/bin/bash

DATASETS="cora citeseer facebook mutag proteins enzymes twitter"

for ds in $DATASETS; do
    echo "=========================================="
    echo " Running baselines: $ds"
    echo "=========================================="
    sed -i '' "s/DATASET    = .*/DATASET    = \"$ds\"/" main.py
    sed -i '' "s/RUNS       = .*/RUNS       = 10/" main.py
    python3 main.py

    echo "=========================================="
    echo " Running variants: $ds"
    echo "=========================================="
    sed -i '' "s/DATASET    = .*/DATASET    = \"$ds\"/" run_variants.py
    sed -i '' "s/RUNS       = .*/RUNS       = 10/" run_variants.py
    python3 run_variants.py
done

echo "ALL DONE"