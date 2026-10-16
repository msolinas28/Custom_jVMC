"""
CPU correctness check of the dense SR path: SR updates (Pinv, and PinvSNR with the SNR cutoff, which
moves the Jacobian between its states) for a small real RBM, with whatever jVMC_exp is importable.
Run it before and after a change and compare:

    export JAX_PLATFORMS=cpu JVMC_USE_DISTRIBUTED=false XLA_FLAGS=--xla_force_host_platform_device_count=4
    python check_dense.py before.npz                  # current code
    python check_dense.py after.npz --ref before.npz  # changed code: prints the relative differences

With 4 devices the samples (2002) and the parameters (420) are padded, which exercises the padding.
"""
import sys, os, warnings, argparse
parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--ref", default=None, help="npz of a previous run to compare with")
parser.add_argument("--n", type=int, default=2002)
args = parser.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "benchmark", "netket_comparison"))

import numpy as np, jax, jax.numpy as jnp
import jVMC_exp
from jVMC_exp.nets import RBM
from jVMC_exp.objective_function import Observable
from jVMC_exp.optimizer import SR
from jVMC_exp.sampler import MCSampler
from jVMC_exp.sampler.propose import SpinFlip
from jVMC_exp.solver import Pinv, PinvSNR
from jVMC_exp.vqs import NQS
from compare_netket import jvmc_hamiltonian

L, H, n = 20, 21, args.n
rng = np.random.default_rng(0)
psi = NQS(RBM(numHidden=H, bias=False), L, batchSize=512, seed=0)
psi.parameters = {"Dense_0": {"kernel": jnp.asarray(rng.normal(0, 1 / np.sqrt(L), (L, H)))}}
ham = jvmc_hamiltonian(L, 1.0)
sampler = MCSampler(psi, SpinFlip(), key=0, numChains=jax.device_count(), numSamples=n)
s = jnp.asarray(rng.integers(0, 2, (n, L)), dtype=jVMC_exp.global_defs.DT_SAMPLES)
sampler._samples, sampler._logPsi, sampler._weights = s, psi(s), jnp.asarray(rng.uniform(0.5, 1.5, n))

out = {}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for name, solver in [("pinv", Pinv()), ("snr", PinvSNR(snr_tol=2.0, pinv_tol=1e-6, pinv_cutoff=1e-10))]:
        sr = SR(sampler, psi, diagonalShift=1e-3, solver=solver)
        for rep in range(2):
            out[f"{name}{rep}"] = np.asarray(sr.get_update(Observable(ham).value_and_grad(sampler)))
donation = [str(w.message) for w in caught if "donat" in str(w.message).lower()]
print(f"{jax.device_count()} devices, params pad {sr._params_pad_size}, donation warnings: {donation or 'none'}")
np.savez(args.out, **out)

if args.ref:
    ref = np.load(args.ref)
    print("relative difference to", args.ref, {k: f"{np.linalg.norm(ref[k] - out[k]) / np.linalg.norm(ref[k]):.1e}" for k in ref.files})
