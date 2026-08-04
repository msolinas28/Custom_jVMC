#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

N_BATCHES_LIST=(1 2 4 8)

echo "=== jacobian=dense  n_batches=1 ==="
python jacobian_memory_sweep.py dense --n_batches 1

for n_batches in "${N_BATCHES_LIST[@]}"; do
    echo "=== jacobian=batched  n_batches=$n_batches ==="
    if ! python jacobian_memory_sweep.py batched --n_batches "$n_batches"; then
        echo "    FAILED (jacobian=batched n_batches=$n_batches) -- continuing"
    fi
done
