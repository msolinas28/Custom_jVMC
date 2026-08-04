import sys
sys.path.append(sys.path[0] + "/../..")
import argparse
import os
import resource
import time
import jax
import pandas as pd

import jVMC_exp
import jVMC_exp.nets as nets
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler
from jVMC_exp.vqs import NQS
parser = argparse.ArgumentParser()
parser.add_argument("jacobian", choices=["dense", "batched"])
parser.add_argument("--n_batches", type=int, default=1)
args = parser.parse_args()

def build_hamiltonian(L, J=-1.0, hx=-0.7):
    H = 0
    for i in range(L):
        H += J * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) + hx * op.SigmaX(i)
    return H

def device_memory():
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return None, None
    return stats.get("bytes_in_use"), stats.get("peak_bytes_in_use")

def host_max_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024

num_hidden = 512
L = 10
n_samples = 2**16
n_chains = n_samples // 16
batch_size = n_samples // args.n_batches
batched = args.jacobian == "batched"

rbm = nets.CpxRBM(numHidden=num_hidden, bias=False)
psi = NQS(rbm, L, batch_size, seed=123)
mc_sampler = sampler.MCSampler(
    psi, jVMC_exp.propose.SpinFlip(), 
    key=123, numChains=n_chains, numSamples=n_samples
)
hamiltonian = build_hamiltonian(L)
loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=batched)

mc_sampler.sample()
o_loc = loss_function(mc_sampler)

if batched:
    grad_log_psi = jVMC_exp.stats.LazySampledObs(mc_sampler.psi.lazy_gradients(mc_sampler.samples), mc_sampler.weights)
else:
    grad_log_psi = jVMC_exp.stats.SampledObs(mc_sampler.psi.gradients(mc_sampler.samples), mc_sampler.weights)
# grad, grad_var_re, grad_var_im, grad_cov_re_im = grad_log_psi.get_covar_and_covar_var(o_loc)
# jax.block_until_ready((grad, grad_var_re, grad_var_im, grad_cov_re_im))
jax.block_until_ready(grad_log_psi.get_covar(o_loc))

# del grad, grad_var_re, grad_var_im, grad_cov_re_im
del grad_log_psi

t0 = time.perf_counter()
if batched:
    grad_log_psi = jVMC_exp.stats.LazySampledObs(mc_sampler.psi.lazy_gradients(mc_sampler.samples), mc_sampler.weights)
else:
    grad_log_psi = jVMC_exp.stats.SampledObs(mc_sampler.psi.gradients(mc_sampler.samples), mc_sampler.weights)
# grad, grad_var_re, grad_var_im, grad_cov_re_im = grad_log_psi.get_covar_and_covar_var(o_loc)
# jax.block_until_ready((grad, grad_var_re, grad_var_im, grad_cov_re_im))
jax.block_until_ready(grad_log_psi.get_covar(o_loc))
t = time.perf_counter() - t0

results = dict(
    jacobian=args.jacobian,
    n_batches=args.n_batches,
    n_params=psi.numParameters,
    n_samples=mc_sampler.numSamples,
    time=t,
    peak_bytes_in_use=device_memory()[1],
    host_max_rss_bytes=host_max_rss_bytes(),
)
df = pd.DataFrame([results])
df_old = pd.read_csv(f"jacobian_test_covar.csv") if os.path.exists(f"jacobian_test_covar.csv") else pd.DataFrame()
df = pd.concat([df_old, df], ignore_index=True)
df.to_csv(f"jacobian_test_covar.csv", index=False)