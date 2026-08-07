# from jVMC_exp.solver.util import diagonalize
# from jVMC_exp.sharding_config import MESH_2D

# import jax
# import jax.numpy as jnp
# import argparse
# import math

# if jax.process_index() == 0:
#     print("Devices =", jax.devices())
#     print("Device count =", jax.device_count())

# parser = argparse.ArgumentParser()
# parser.add_argument("N", type=int)
# args = parser.parse_args()

# N = args.N
# A = jnp.eye(N)

# m = jax.device_count()
# pad_size = (-N) % m
# N_padded = N + pad_size

# c = 2 * jnp.linalg.norm(A) + 1

# A = jnp.pad(A, ((0, pad_size), (0, pad_size)), mode="constant")

# p_rows, p_cols = MESH_2D.shape["row"], MESH_2D.shape["col"]
# T_A = math.gcd((A.shape[0]) // p_rows, (A.shape[0]) // p_cols)

# if jax.process_index() == 0:
#     print(f"N={N} padded to {N_padded} (+{pad_size}), {m} devices -> {p_rows}x{p_cols} grid, T_A={T_A}")

# ev, V = diagonalize(A, pad_size, mode='distributed')

# if jax.process_index() == 0:
#     print('Matrix original size: ', N)
#     print(f'Eigenvalues: \n Shape={ev.shape} \n Shards={ev.addressable_shards}')
#     print(f'Eigenvectors: \n Shape={V.shape} \n Shards={V.addressable_shards}')

import jVMC_exp
import jax

L = 19

n_samples = 2**12
n_chains = n_samples // 4
batch_size = n_samples

net = jVMC_exp.nets.RBM(L)
psi = jVMC_exp.vqs.NQS(net, L, batch_size, seed=123)
sampler = jVMC_exp.sampler.MCSampler(
    psi, 
    jVMC_exp.propose.SpinFlip(),
    123,
    n_chains, 
    n_samples
)

if jax.process_index() == 0:
    print("# parameters:", psi.numParameters)
    print("Holo:", psi.holomorphic)
    print("Real params:", psi.realParams)

J = -1
h = 0.5
H = 0
for i in range(L):
    H += J * jVMC_exp.operator.discrete.SigmaZ(i) * jVMC_exp.operator.discrete.SigmaZ((i + 1) % L)
    H += h * jVMC_exp.operator.discrete.SigmaX(i)

loss_function = jVMC_exp.objective_function.Observable(H)
solver = jVMC_exp.solver.PinvSNR(diagonalization_mode="distributed")
stepper = jVMC_exp.stepper.Euler(1e-2)
opt = jVMC_exp.optimizer.SR(sampler, psi, solver=solver)

psi.parameters = opt.step(0, stepper, loss_function)[0]

if jax.process_index() == 0:
    print("Done")