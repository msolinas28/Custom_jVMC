import os, sys, json, time, subprocess
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

LX = 20                 # 20x20 TFIM -> L=400, N_conn=401
OP_BATCH = 32           # H.batch_size (samples per fused batch)
PSI_BATCH = 1024
N_SAMPLES = (512, 1024, 2048, 4096, 8192)


def peak_bytes():
    import jax
    stats = jax.local_devices()[0].memory_stats()
    if stats is not None:                      # GPU: allocator peak
        return stats["peak_bytes_in_use"]
    import resource                            # CPU fallback: process peak RSS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def run_one(method, Lx, n_samples, op_batch):
    import jax, jax.numpy as jnp
    import jVMC_exp
    import jVMC_exp.operator as op

    L = Lx * Lx
    H = 0
    for i in range(L):
        x, y = divmod(i, Lx)
        H += op.discrete.SigmaZ(i) * op.discrete.SigmaZ(((x + 1) % Lx) * Lx + y)
        H += op.discrete.SigmaZ(i) * op.discrete.SigmaZ(x * Lx + (y + 1) % Lx)
        H += op.discrete.SigmaX(i)
    H.batch_size = op_batch

    psi = jVMC_exp.vqs.NQS(jVMC_exp.nets.RBM(L), L, batchSize=PSI_BATCH)
    s = jax.random.randint(jax.random.PRNGKey(0), (n_samples, L), 0, 2, dtype=jnp.int32)
    log_psi = psi(s).block_until_ready()
    before = peak_bytes()

    f = H.get_O_loc if method == "old" else H.get_O_loc_new
    eloc = f(s, psi, logPsiS=log_psi).block_until_ready()      # 1st call: compile + run
    t0 = time.perf_counter()
    eloc = f(s, psi, logPsiS=log_psi).block_until_ready()      # 2nd call: timing only
    dt = time.perf_counter() - t0

    return dict(peak_gb=peak_bytes() / 1e9, before_gb=before / 1e9,
                time_s=dt, eloc_mean=repr(complex(jnp.mean(eloc))))


def run_child(method, n):
    out = subprocess.run(
        [sys.executable, __file__, "child", method, str(LX), str(n), str(OP_BATCH)],
        capture_output=True, text=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    tail = (out.stderr.strip().splitlines() or ["no output"])[-1]
    return dict(error=f"exit {out.returncode}: {tail[:100]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        _, _, method, Lx, n, b = sys.argv
        try:
            res = run_one(method, int(Lx), int(n), int(b))
        except Exception as e:  # XLA OOM raises a runtime error
            res = dict(error=f"{type(e).__name__}: {str(e).splitlines()[0][:100]}")
        print("RESULT " + json.dumps(res))
        sys.exit()

    L = LX * LX
    print(f"TFIM {LX}x{LX}, L={L}, N_conn={L + 1}, H.batch_size={OP_BATCH}, psi.batchSize={PSI_BATCH}")
    print(f"{'N':>6} {'s_p size':>9} | {'old peak':>9} {'old t':>7} | {'new peak':>9} {'new t':>7} | E_loc mean old / new")
    for n in N_SAMPLES:
        sp_gb = n * (L + 1) * L * 4 / 1e9
        r = {m: run_child(m, n) for m in ("old", "new")}
        cols = []
        for m in ("old", "new"):
            if "error" in r[m]:
                cols.append(f"{'FAIL':>9} {'':>7}")
            else:
                cols.append(f"{r[m]['peak_gb']:8.2f}G {r[m]['time_s']:6.2f}s")
        e = " / ".join(r[m].get("eloc_mean", "-") for m in ("old", "new"))
        print(f"{n:>6} {sp_gb:8.2f}G | {cols[0]} | {cols[1]} | {e}")
        for m in ("old", "new"):
            if "error" in r[m]:
                print(f"         {m}: {r[m]['error']}")