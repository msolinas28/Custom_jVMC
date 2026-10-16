"""
Where the peak device memory of the jVMC_exp dense SR step comes from (see ../netket_sr_comparison.md).
Uses the setup of tests/benchmark/netket_comparison/compare_netket_sr.py (run_jvmc / run_netket),
one process per point, since the peak memory of a process can't be reset.

    python probe_sr_memory.py stages  --n 131072 --out OUT   # dense path, memory after every stage
    python probe_sr_memory.py dense   --n 524288 --out OUT   # dense path, as in the benchmark
    python probe_sr_memory.py blocked --n 524288 --out OUT   # old prototype: S and force from row blocks (monkeypatch)
    python probe_sr_memory.py netket  --n 262144 --out OUT   # NetKet, as in the benchmark

OUT must contain config.json and params.npz of a compare_netket_sr.py run (the batch size of the
run is the one in config.json). The probe writes probe_<mode><tag>_<n>.json and the update into OUT,
and compares the update with OUT/ref/update_<series>_<n>.npy if it exists.
XLA_FLAGS=--xla_gpu_autotune_level=0 removes the cuBLAS/fusion autotuning buffers.
"""
import sys, os, json, argparse

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "benchmark", "netket_comparison")
sys.path.insert(0, BENCH)

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["stages", "dense", "blocked", "netket"])
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tag", default="")
args = parser.parse_args()

import numpy as np
import compare_netket_sr as C

with open(os.path.join(args.out, "config.json")) as f:
    cfg = json.load(f)
P = cfg["L"] * cfg["num_hidden"] + (cfg["num_hidden"] if cfg["bias"] else 0)
J_BYTES = args.n * P * 8
LOG = []
CALL = [0]

def mem():
    import jax
    s = jax.local_devices()[0].memory_stats()
    return (s["bytes_in_use"], s["peak_bytes_in_use"]) if s else (0, 0)

def recorded(name, fn):
    import jax
    def wrapped(*a, **k):
        out = fn(*a, **k)
        jax.block_until_ready(out)
        used, peak = mem()
        shape = next((tuple(x.shape) for x in a if hasattr(x, "shape") and x.ndim == 2), None)
        LOG.append(dict(call=CALL[0], stage=name, shape=shape, used=used, peak=peak))
        return out
    return wrapped

def instrument():
    import jVMC_exp.stats as stats
    from jVMC_exp.optimizer.base import Evolution
    from jVMC_exp.objective_function.base import Observable
    from jVMC_exp.sampler.mc_sampler import MCSampler
    from jVMC_exp.vqs import NQS
    from jVMC_exp.solver import Pinv

    vag = Observable.value_and_grad
    def counted(self, *a, **k):
        CALL[0] += 1
        return vag(self, *a, **k)
    Observable.value_and_grad = counted
    MCSampler.__call__ = recorded("E_loc", MCSampler.__call__)
    NQS.gradients = recorded("jacobian", NQS.gradients)
    stats._get_weighted_covar = recorded("force", stats._get_weighted_covar)
    stats._normalize = recorded("normalize", stats._normalize)
    if hasattr(stats, "_normalize_in_place"):
        stats._normalize_in_place = recorded("normalize", stats._normalize_in_place)
    Evolution._add_qgt = recorded("add_qgt", Evolution._add_qgt)
    Evolution._get_lhs_dense = recorded("lhs+shift", Evolution._get_lhs_dense)
    Pinv._diagonalize = recorded("eigh+V^Tb", Pinv._diagonalize)
    Pinv.__call__ = recorded("solver", Pinv.__call__)

def apply_blocked_fix():
    """
    Prototype: the dense path keeps the stored Jacobian J, but builds the force and S from
    row blocks of batchSize samples of it, as the batched path does with recomputed blocks.
    No normalized copy of J, and every GEMM has the shape of one block.
    """
    import jVMC_exp.stats as stats
    from jVMC_exp.stats import SampledObs
    from jVMC_exp.sharding_config import BatchLayout
    from jVMC_exp.optimizer import base
    from jVMC_exp.objective_function.base import Observable, ObjectiveFunctionOutput

    def layout_of(obs, batch_size):
        n = obs.observations.shape[0]
        return BatchLayout(n, min(batch_size, n))

    def block(x, layout, i):
        return layout.trim(layout.take(x, i), i)

    vag = Observable.value_and_grad
    def value_and_grad(self, sampler, compute_grad=True, **op_kwargs):
        if self._batched_jacobian or not compute_grad:
            return vag(self, sampler, compute_grad, **op_kwargs)
        o_loc = self(sampler, **op_kwargs)
        obs = SampledObs(sampler.psi.gradients(sampler.samples), sampler.weights)
        layout = layout_of(obs, sampler.psi.batchSize)
        grad = 0
        for i, (w, other) in enumerate(zip(layout.split(obs.weights), layout.split(o_loc._normalized_obs))):
            grad += stats._get_weighted_covar(block(obs.observations, layout, i), w, other)
        return ObjectiveFunctionOutput(o_loc=o_loc, grad=grad, grad_log_psi=obs)
    Observable.value_and_grad = value_and_grad

    lhs = base.Evolution._get_lhs_dense
    def get_lhs_dense(self, grad_log_psi):
        if not isinstance(grad_log_psi, SampledObs):
            return lhs(self, grad_log_psi)
        J, mean = grad_log_psi.observations, grad_log_psi.mean
        layout = layout_of(grad_log_psi, self.psi.batchSize)
        S = None
        for i, w in enumerate(layout.split(grad_log_psi.weights)):
            b = stats._normalize(block(J, layout, i), w, mean)
            if self._params_pad_size != 0:
                b = self._pad_obs(b)
            if S is None:
                S = base._init_qgt(b)
            S = self._add_qgt(S, b, batch_size=None)
            del b
        del J
        if self.store_qgt:
            self._qgt = S
        S = self._lhs_trans_fn(S)
        if self.diag_scale > 1e-15:
            idx = base.jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].multiply(1 + self.diag_scale)
        if self.diag_shift > 1e-15:
            idx = base.jnp.arange(S.shape[0] - self._params_pad_size)
            S = S.at[idx, idx].add(self.diag_shift)
        return S
    base.Evolution._get_lhs_dense = get_lhs_dense
    # _get_lhs is bound in __init__, so instances created later pick up the new method

if args.mode == "stages":
    instrument()
if args.mode == "blocked":
    apply_blocked_fix()

series = "netket" if args.mode == "netket" else "jvmc-dense"
try:
    if args.mode == "netket":
        result = C.run_netket(cfg, args.n, args.out)
    else:
        result = C.run_jvmc(cfg, args.n, args.out, batched=False)
except Exception as error:
    if not C.is_out_of_memory(error):
        raise
    result = {"oom": True, "error": str(error)[:2000]}

name = f"probe_{args.mode}{args.tag}_{args.n}"
result.update(mode=args.mode, n_samples=args.n, J_bytes=J_BYTES, xla_flags=os.environ.get("XLA_FLAGS", ""), log=LOG)
with open(os.path.join(args.out, name + ".json"), "w") as f:
    json.dump(result, f, indent=1)

# Same update as the stored reference of the benchmark run?
ref = os.path.join(args.out, "ref", f"update_{series}_{args.n}.npy")
new = os.path.join(args.out, f"update_{series}_{args.n}.npy")
if os.path.exists(ref) and os.path.exists(new):
    a, b = np.load(ref), np.load(new)
    print(f"relative difference to the reference update: {np.linalg.norm(a - b) / np.linalg.norm(a):.2e}")

J = J_BYTES
if "times" in result:
    print(f"{name}: SR {np.median(result['times']['sr_step']):.3f} s  E_loc {np.median(result['times']['local_energies']):.3f} s  "
          f"first SR {result['first_call']['sr_step']:.1f} s  peak {result['peak_bytes'] / 1e9:.2f} GB = {result['peak_bytes'] / J:.2f} J  (J = {J / 1e9:.2f} GB)")
else:
    print(f"{name}: OUT OF MEMORY: {result['error'][:300]}")
prev = 0
for e in LOG:
    jump = e["peak"] - prev
    prev = max(prev, e["peak"])
    print(f"  call {e['call']} {e['stage']:>10} {str(e['shape']):>16}  in use {e['used'] / J:5.2f} J   peak {e['peak'] / J:5.2f} J"
          + (f"   <- peak +{jump / J:.2f} J" if jump > 0.05 * J else ""))
