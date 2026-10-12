#!/bin/bash -l
#SBATCH --account=neuquass
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1            # The comparison runs on a single GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # Used by the Numba operators of NetKet, if requested
#SBATCH --time=01:00:00
#SBATCH --output=%j.out

module --force purge
module load Stages/2026
module load GCCcore/.14.3.0
module load CUDA/13
module load Python/3.13.5
module load cuDNN/9.19.0.56-CUDA-13
module load NCCL/default-CUDA-13

# NetKet has to be installed in the same environment (pip install netket), so that
# both libraries run with the same JAX
source "$SCRATCH"/env/jVMC/bin/activate

# Usage: sbatch send_job.sh [compare_netket.py | compare_netket_sr.py] [options of the script]
srun python -u "${1:-compare_netket.py}" "${@:2}"
