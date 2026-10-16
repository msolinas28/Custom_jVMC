"""
CPU correctness check of MinSR: updates with the dense and the batched Jacobian, for a real RBM
and a holomorphic CpxRBM, with whatever jVMC_exp is importable. Run it before and after a change:

    export JAX_PLATFORMS=cpu JVMC_USE_DISTRIBUTED=false XLA_FLAGS=--xla_force_host_platform_device_count=4
    python check_minsr.py before.npz
    python check_minsr.py after.npz --ref before.npz

With 4 devices the samples (1002, batches of 256) and the parameters (441) are padded.
"""
import sys, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--ref", default=None)
args = parser.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "benchmark", "netket_comparison"))

import numpy as np, jax, jax.numpy as jnp
import jVMC_exp
from jVMC_exp.nets import RBM, CpxRBM
from jVMC_exp.objective_function import Observable
from jVMC_exp.optimizer import MinSR
from jVMC_exp.sampler import MCSampler
from jVMC_exp.sampler.propose import SpinFlip
from jVMC_exp.solver import Pinv
from jVMC_exp.vqs import NQS
from compare_netket import jvmc_hamiltonian

L, H, n = 21, 21, 1002
ham = jvmc_hamiltonian(L, 1.0)
out = {}
for net_name, net in (("rbm", RBM(numHidden=H, bias=False)), ("cpxrbm", CpxRBM(numHidden=H, bias=False))):
    rng = np.random.default_rng(0)
    psi = NQS(net, L, batchSize=256, seed=0)
    sampler = MCSampler(psi, SpinFlip(), key=0, numChains=jax.device_count(), numSamples=n)
    s = jnp.asarray(rng.integers(0, 2, (n, L)), dtype=jVMC_exp.global_defs.DT_SAMPLES)
    sampler._samples, sampler._logPsi, sampler._weights = s, psi(s), jnp.full(n, 1 / n)
    minsr = MinSR(sampler, psi, diagonalShift=1e-3, solver=Pinv())
    for batched in (False, True):
        for rep in range(2):
            update = minsr.get_update(Observable(ham, batched_jacobian=batched).value_and_grad(sampler, compute_grad=False))
            out[f"{net_name}_{'batched' if batched else 'dense'}{rep}"] = np.asarray(update)
print(f"{jax.device_count()} devices, params pad {minsr._params_pad_size}")
np.savez(args.out, **out)

if args.ref:
    ref = np.load(args.ref)
    print("relative difference to", os.path.basename(args.ref),
          {k: f"{np.linalg.norm(ref[k] - out[k]) / np.linalg.norm(ref[k]):.1e}" for k in ref.files})
