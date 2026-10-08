#!/usr/bin/env bash
# Runs minsr_memory_probe2.py in every setting, one process each (the peak memory counter
# cannot be reset). Run it on a GPU node once per branch: the scripts and the results are not
# tracked by git, so they survive `git checkout`. The label defaults to the current branch.
set -uo pipefail
cd "$(dirname "$0")"

# Branch read from .git/HEAD, git is not installed on the compute nodes
BRANCH="$(sed -n 's|^ref: refs/heads/||p' ../../../.git/HEAD 2>/dev/null)"
LABEL="${1:-${BRANCH:-unknown}}"
PYTHON="${PYTHON:-python}"
# Every run writes to a new folder, so that results of different runs are never mixed
OUT_DIR="probe2_results/${LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

run() {
    local prealloc=$1; shift
    local out="$OUT_DIR/prealloc-${prealloc}_$(echo "$*" | tr -d '-' | tr ' ' '_').txt"
    # 'false' is what the benchmark sets for its cases, 'default' is how the first probe ran
    local env_args=(-u XLA_PYTHON_CLIENT_PREALLOCATE)
    [ "$prealloc" = false ] && env_args=(XLA_PYTHON_CLIENT_PREALLOCATE=false)

    echo "=== prealloc=$prealloc $* ==="
    if ! env "${env_args[@]}" "$PYTHON" -u minsr_memory_probe2.py "$@" > "$out" 2>&1; then
        echo "    FAILED -- continuing, see $out"
    fi
    grep -v cuda_vmm_allocator "$out" | grep "^rep \|^RESULT"
}

for prealloc in false default; do
    for sync in none grad obs step all; do
        run "$prealloc" --case dense --sync "$sync"
    done
    for sync in none step; do
        run "$prealloc" --case lazy --sync "$sync"
    done
done

echo; echo "=== summary ($LABEL) ==="
grep -h "^RESULT" "$OUT_DIR"/*.txt