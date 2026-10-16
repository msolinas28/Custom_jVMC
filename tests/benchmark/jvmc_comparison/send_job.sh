#!/bin/bash -l
#SBATCH --account=neuquass
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1            # The comparison runs on a single GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%j.out

module --force purge
module load Stages/2026
module load GCCcore/.14.3.0
module load CUDA/13
module load Python/3.13.5
module load cuDNN/9.19.0.56-CUDA-13
module load NCCL/default-CUDA-13
# The original jVMC imports mpi4py. Loaded for both libraries, so that both run in the same environment
module load GCC/14.3.0 ParaStationMPI/5.13.0-1 mpi4py/4.1.0

# Both libraries run with the same Python environment, i.e. the same JAX
source "$SCRATCH"/env/jVMC/bin/activate

# Folder with the package jVMC of the original library (see setup_original.sh)
JVMC_ORIGINAL="${JVMC_ORIGINAL:-$SCRATCH/env/jVMC_original}"

# Usage: sbatch send_job.sh [compare_jvmc.py | compare_jvmc_sr.py | compare_jvmc_minsr.py] [options of the script]
srun python -u "${1:-compare_jvmc.py}" --jvmc-path "$JVMC_ORIGINAL" "${@:2}"
