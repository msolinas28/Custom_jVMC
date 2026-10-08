export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0
OUT=lazy_probe_$(sed -n 's|^ref: refs/heads/||p' ../../../.git/HEAD)
mkdir -p $OUT
for sync in none step batches norm; do
    python -u minsr_memory_probe2.py --case lazy --sync $sync > $OUT/sync_$sync.txt 2>&1
    grep "^RESULT" $OUT/sync_$sync.txt
done