"""
Where the peak device memory of the jVMC_exp MinSR step comes from: the MinSR version of
probe_sr_memory.py. Uses the setup of tests/benchmark/netket_comparison/compare_netket_minsr.py
(run_jvmc / run_netket), one process per point.

    python probe_minsr_memory.py stages  --hidden 4096 --out OUT             # dense Jacobian, memory after every stage
    python probe_minsr_memory.py stages  --hidden 4096 --out OUT --batched   # batched Jacobian
    python probe_minsr_memory.py dense   --hidden 8192 --out OUT             # as in the benchmark (also --batched)
    python probe_minsr_memory.py netket  --hidden 4096 --out OUT [--on-the-fly]

OUT must contain config.json and params_<hidden>.npz of a compare_netket_minsr.py run (the number
of samples and the batch size are the ones in config.json). J = n_samples * n_parameters * 8 bytes.
XLA_FLAGS=--xla_gpu_autotune_level=0 removes the autotuning buffers.
"""
import sys, os, json, argparse

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "benchmark", "netket_comparison")
sys.path.insert(0, BENCH)

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["stages", "dense", "netket"])
parser.add_argument("--hidden", type=int, required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--batched", action="store_true")
parser.add_argument("--on-the-fly", action="store_true")
parser.add_argument("--tag", default="")
args = parser.parse_args()

import numpy as np
import compare_netket_minsr as M

with open(os.path.join(args.out, "config.json")) as f:
    cfg = json.load(f)
P = M.n_parameters(cfg, args.hidden)
J_BYTES = cfg["n_samples"] * P * 8
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
    import jVMC_exp.optimizer.minsr as minsr
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
    stats._get_mean = recorded("mean", stats._get_mean)
    for name in ("_normalize", "_normalize_in_place"):
        if hasattr(stats, name):
            setattr(stats, name, recorded("normalize", getattr(stats, name)))
    minsr._normalize_batch = recorded("norm_batch", minsr._normalize_batch)
    minsr._adjoint_dot = recorded("O^T x", minsr._adjoint_dot)
    minsr.MinSR._get_single_tangent_kernel = recorded("T single", minsr.MinSR._get_single_tangent_kernel)
    minsr.MinSR._get_double_tangent_kernel = recorded("T double", minsr.MinSR._get_double_tangent_kernel)
    Pinv.__call__ = recorded("solver", Pinv.__call__)

if args.mode == "stages":
    instrument()

if args.mode == "netket":
    series = "netket-onthefly" if args.on_the_fly else "netket"
else:
    series = "jvmc-batched" if args.batched else "jvmc-dense"
try:
    if args.mode == "netket":
        result = M.run_netket(cfg, args.hidden, args.out, args.on_the_fly)
    else:
        result = M.run_jvmc(cfg, args.hidden, args.out, args.batched)
except Exception as error:
    if not M.is_out_of_memory(error):
        raise
    result = {"oom": True, "error": str(error)[:2000]}

name = f"probe_{args.mode}_{series}{args.tag}_{args.hidden}"
result.update(mode=args.mode, series=series, num_hidden=args.hidden, J_bytes=J_BYTES,
              xla_flags=os.environ.get("XLA_FLAGS", ""), log=LOG)
with open(os.path.join(args.out, name + ".json"), "w") as f:
    json.dump(result, f, indent=1)

ref = os.path.join(args.out, "ref", f"update_jvmc-dense_{args.hidden}.npy")
new = os.path.join(args.out, f"update_{series}_{args.hidden}.npy")
if os.path.exists(ref) and os.path.exists(new):
    a, b = np.real(np.load(ref)), np.real(np.load(new))
    factor = M.NETKET_UPDATE_FACTOR if series.startswith("netket") else 1
    print(f"relative difference to the reference update: {np.linalg.norm(b - factor * a) / np.linalg.norm(b):.2e}")

J = J_BYTES
if "times" in result:
    t = np.median(result["times"]["minsr_step"])
    print(f"{name}: MinSR {t:.3f} s  E_loc {np.median(result['times']['local_energies']):.3f} s  first {result['first_call']['minsr_step']:.1f} s  "
          f"peak {result['peak_bytes'] / 1e9:.2f} GB = {result['peak_bytes'] / J:.2f} J  (J = {J / 1e9:.2f} GB, P = {P}, N = {cfg['n_samples']})")
else:
    print(f"{name}: OUT OF MEMORY: {result['error'][:300]}")
prev = 0
for e in LOG:
    jump = e["peak"] - prev
    prev = max(prev, e["peak"])
    print(f"  call {e['call']} {e['stage']:>10} {str(e['shape']):>16}  in use {e['used'] / J:5.2f} J   peak {e['peak'] / J:5.2f} J"
          + (f"   <- peak +{jump / J:.2f} J" if jump > 0.05 * J else ""))
