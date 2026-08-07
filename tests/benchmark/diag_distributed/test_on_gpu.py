from jVMC_exp.solver.util import diagonalize
from jVMC_exp.sharding_config import MESH_2D

import jax
import jax.numpy as jnp
import argparse
import math

if jax.process_index() == 0:
    print("Devices =", jax.devices())
    print("Device count =", jax.device_count())

parser = argparse.ArgumentParser()
parser.add_argument("N", type=int)
args = parser.parse_args()

N = args.N
A = jnp.eye(N)

m = jax.device_count()
pad_size = (-N) % m
N_padded = N + pad_size

c = 2 * jnp.linalg.norm(A) + 1

A = jnp.pad(A, ((0, pad_size), (0, pad_size)), mode="constant")

p_rows, p_cols = MESH_2D.shape["row"], MESH_2D.shape["col"]
T_A = math.gcd((A.shape[0]) // p_rows, (A.shape[0]) // p_cols)

if jax.process_index() == 0:
    print(f"N={N} padded to {N_padded} (+{pad_size}), {m} devices -> {p_rows}x{p_cols} grid, T_A={T_A}")

ev, V = diagonalize(A, pad_size, mode='distributed')

if jax.process_index() == 0:
    print('Matrix original size: ', N)
    print(f'Eigenvalues: \n Shape={ev.shape} \n Shards={ev.addressable_shards}')
    print(f'Eigenvectors: \n Shape={V.shape} \n Shards={V.addressable_shards}')