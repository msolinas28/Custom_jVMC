"""ProjectedOrbitNet inside NQS: exact projection, 2D sample shapes, direct sampling, construction time."""
import sys, time, itertools
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
from symm_dense import make_basis, dense_P
import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
from jVMC_exp.symmetry import ProjectedOrbitNet
from jVMC_exp.symmetry.lattice_symetries import (square_translation_symmetry, square_d4_symmetry,
                                                 spin_flip_symmetry, particle_hole_symmetry, point_group_symmetry)

def run(name, fn):
    try:
        print(f"[{name}]", fn())
    except Exception as e:
        print(f"[{name}] EXC {type(e).__name__}: {str(e)[:200]}")

nd = jax.device_count()
# a) exp average = exact projection P psi
L = 6
b = make_basis(L, 2)
for kind in ("spin", "spinless_fermion"):
    symm = square_translation_symmetry(L, 1, kind, q=(1, 0))
    base = NQS(nets.CpxRBM(numHidden=4, bias=True), L, batchSize=8 * nd, seed=3)
    proj = NQS(nets.CpxRBM(numHidden=4, bias=True), L, batchSize=8 * nd, seed=3, orbit=symm)
    proj.parameters = base.parameters_flat
    psi = np.asarray(jnp.exp(base(jnp.asarray(b))))
    psi_p = np.asarray(jnp.exp(proj(jnp.asarray(b))))
    run(f"a exact projection {kind}", lambda: bool(np.allclose(psi_p, dense_P(symm, b) @ psi)))

# b) two-dimensional sample shape with a 2D network
symm = square_translation_symmetry(3, 3, "spin")
run("b CNN, sampleShape (3, 3), no orbit", lambda: NQS(nets.CNN(F=(3, 3), channels=(2,), strides=(1, 1)), (3, 3), batchSize=9 * nd, seed=1)(jnp.zeros((9 * nd, 3, 3), jnp.int32)).shape)
run("b CNN, sampleShape (3, 3), orbit", lambda: NQS(nets.CNN(F=(3, 3), channels=(2,), strides=(1, 1)), (3, 3), batchSize=9 * nd, seed=1, orbit=symm)(jnp.zeros((9 * nd, 3, 3), jnp.int32)).shape)

# c) direct sampling from a projected generator with 'sep': histogram against exact |psi_P|^2
import flax.linen as nn
class ProductGenerator(nn.Module):
    L: int
    def setup(self):
        self.a = self.param("a", lambda k, sh: jnp.linspace(-1.5, 2.0, self.L), (self.L,))
    def __call__(self, s):
        logp = jnp.where(s == 1, jax.nn.log_sigmoid(self.a), jax.nn.log_sigmoid(-self.a))
        return 0.5 * jnp.sum(logp) + 0.3j * jnp.sum(s * jnp.arange(self.L))
    def sample(self, key):
        return jax.random.bernoulli(key, jax.nn.sigmoid(self.a)).astype(jnp.int32)
L = 4
b = make_basis(L, 2)
for name, symm in [("spin flip", spin_flip_symmetry(L, 1)), ("translation", square_translation_symmetry(L, 1, "spin"))]:
    psi = NQS(ProductGenerator(L), L, batchSize=64 * nd, seed=5, orbit=symm, symmetry_average="sep")
    p_exact = np.exp(2 * np.real(np.asarray(psi(jnp.asarray(b)))))
    n = 64 * nd * 500
    s = np.asarray(psi.sample(n, key=jax.random.PRNGKey(0)))
    counts = np.bincount(s @ (2 ** np.arange(L)[::-1]), minlength=2 ** L) / n
    print(f"[c {name}] sum |psi_P|^2 = {p_exact.sum():.6f}, max |freq - p| = {np.abs(counts - p_exact).max():.4f}, "
          f"noise ~ {np.sqrt(p_exact.max() / n):.4f}")

# d) construction time
for Lx in (6, 10, 14):
    t = time.perf_counter(); square_translation_symmetry(Lx, Lx, "spin"); t1 = time.perf_counter() - t
    t = time.perf_counter(); square_d4_symmetry(Lx, Lx, "spin"); t2 = time.perf_counter() - t
    print(f"[d] {Lx}x{Lx}: translation {t1:.1f} s, D4 {t2:.1f} s")

# e) output dtype of a projected net with single-precision parameters
symm = square_translation_symmetry(4, 1, "spin")
pn = ProjectedOrbitNet(base_net=nets.CpxRBM(numHidden=2), symmetry=symm)
v = pn.init(jax.random.PRNGKey(0), jnp.zeros(4, jnp.int32))
run("e output dtype", lambda: pn.apply(v, jnp.zeros(4, jnp.int32)).dtype)
