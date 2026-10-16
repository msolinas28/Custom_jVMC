#!/bin/bash -l
#SBATCH --account=neuquass
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=probe_%j.out
#
# Next memory probe (see ../netket_sr_comparison.md, "Next steps" 1). Submit from this folder:
#     cd Custom_jVMC/.claude/netket_sr_comparison && sbatch probe_job.sh
# or run the same lines in an interactive allocation with one GPU.

module --force purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13 Python/3.13.5 cuDNN/9.19.0.56-CUDA-13 NCCL/default-CUDA-13
# jax 0.11.2, flax 0.12.10, netket 3.22.4 and jVMC_exp installed in editable mode
source "${ENV:-$SCRATCH/env/nk_jvmc}/bin/activate"
export JVMC_USE_DISTRIBUTED=false
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

BENCH=../../tests/benchmark/netket_comparison
NOAT=--xla_gpu_autotune_level=0
p() { echo; echo "######## $*"; srun python -u probe_sr_memory.py "$@" 2>&1 | grep -vE "Warning|warn\(|absl|cudart_stub"; }

# Inputs of the probe (config.json + params.npz) for both batch sizes, from one cheap point of the
# benchmark. --batch-size 0 means all samples in one batch, as in results_sr_20261014_172839.
for b in 0 16384; do
    [ -f out_batch$b/config.json ] || srun python -u $BENCH/compare_netket_sr.py \
        --series jvmc-dense --n-samples 1024 --batch-size $b --out-dir $PWD/out_batch$b > /dev/null
done

# 1. Which stage sets the peak: all samples in one batch, and batches of 16384 samples
p stages --n 131072 --out out_batch0
p stages --n 131072 --out out_batch16384
# 2. The same without autotuning: the peak of the steady state
XLA_FLAGS=$NOAT p stages --n 131072 --out out_batch0 --tag _noautotune
# 3. The point that runs out of memory, with the full error (which instructions the autotuner failed on)
p dense --n 524288 --out out_batch0
p dense --n 524288 --out out_batch16384
