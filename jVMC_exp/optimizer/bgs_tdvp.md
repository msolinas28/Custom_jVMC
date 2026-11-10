# Block Gauss–Seidel TDVP: solving SR without inverting $S$

## 1. Idea

In tVMC/TDVP each time step (or Runge–Kutta stage) requires solving

$$S\,\dot\theta = F,\qquad S\in\mathbb{C}^{N_p\times N_p}.$$

A direct solve costs $O(N_sN_p^2+N_p^3)$ (SR); the MinSR/kernel form costs $O(N_s^2N_p+N_s^3)$. When both $N_s$ and $N_p$ are large, neither is affordable.

The proposal combines two ideas from neural PDE solvers (randomized sparse Neural Galerkin, TENG):

- Partition the parameters into $K$ disjoint blocks $P_1,\dots,P_K$ with $\bigcup_k P_k = P$ and $|P_k| = p_k \ll N_p$.
- Update one block at a time with the others held at their *current* values.
- Sweep over the blocks repeatedly until a stopping criterion is met.

This is block Gauss–Seidel on the TDVP linear system. Only $p_k\times p_k$ matrices are ever inverted, and the coupling between blocks is carried exclusively by a residual vector of length $N_s$.

- One block, one sweep: randomized sparse TDVP (Berman & Peherstorfer).
- Converged sweeps: full SR solution.
- Early-stopped sweeps: a regularized solution in between (Sec. 6).

## 2. Notation

All quantities are evaluated at fixed $\theta$ and on a fixed set of samples $\{x_\alpha\}_{\alpha=1}^{N_s}\sim|\psi_\theta|^2$.

- $O_{\alpha,i} = \dfrac{1}{\sqrt{N_s}}\big(O_i(x_\alpha)-\langle O_i\rangle\big)$: centred log-derivatives, with $O_i(x)=\partial_{\theta_i}\log\psi_\theta(x)$.
- $e_\alpha$: centred, $1/\sqrt{N_s}$-normalized local-energy vector. Its phase convention, e.g. $-i(E_\text{loc}-\langle E\rangle)/\sqrt{N_s}$ in real time, is absorbed into $e$.
- $S_{ij}=\sum_\alpha O^*_{\alpha,i}O_{\alpha,j}$, i.e. $S=O^\dagger O$.
- $F_i=\sum_\alpha O^*_{\alpha,i}\,e_\alpha$, i.e. $F=O^\dagger e$.

TDVP is the normal equation of the least-squares problem

$$\dot\theta=\arg\min_{\dot\theta}\;\|O\dot\theta-e\|^2 .$$

From here on $k$, $i$, $j$ label **blocks**: $\dot\theta_k\in\mathbb{C}^{p_k}$, $O_k\in\mathbb{C}^{N_s\times p_k}$ are the columns of block $k$, and $S_{kj}=O_k^\dagger O_j$.

## 3. Derivation

### 3.1 Block row of the TDVP equation

Row block $k$ of $S\dot\theta=F$ reads

$$\sum_{i}S_{ki}\dot\theta_i = F_k
\quad\Longrightarrow\quad
S_{kk}\dot\theta_k = F_k-\sum_{i\neq k}S_{ki}\dot\theta_i .$$

Inserting $S_{ki}=\sum_\alpha O^*_{\alpha,k}O_{\alpha,i}$ and $F_k=\sum_\alpha O^*_{\alpha,k}e_\alpha$:

$$S_{kk}\dot\theta_k=\sum_\alpha O^*_{\alpha,k}\Big(e_\alpha-\sum_{i\neq k}O_{\alpha,i}\dot\theta_i\Big). \tag{1}$$

### 3.2 The right-hand side contains unknowns, so iterate

The $\dot\theta_i$ with $i\neq k$ are unknown, so Eq. (1) is not closed. Insert the current best guess for the other blocks, solve for block $k$, and move on to the next block. Blocks already visited in the current sweep enter with their new values (Gauss–Seidel):

$$S_{kk}\dot\theta_k^{\text{new}}=\sum_\alpha O^*_{\alpha,k}\Big(e_\alpha-\sum_{i\neq k}O_{\alpha,i}\dot\theta_i^{\text{cur}}\Big). \tag{2}$$

### 3.3 Introduce the full residual

Define the residual over **all** blocks, including $k$:

$$r_\alpha \equiv e_\alpha-\sum_{i}O_{\alpha,i}\dot\theta_i^{\text{cur}},\qquad r = e - O\dot\theta^{\text{cur}}\in\mathbb{C}^{N_s}. \tag{3}$$

The bracket in Eq. (2) is $r$ with the contribution of block $k$ added back:

$$e_\alpha-\sum_{i\neq k}O_{\alpha,i}\dot\theta_i^{\text{cur}} = r_\alpha+O_{\alpha,k}\dot\theta_k^{\text{cur}} .$$

Hence

$$S_{kk}\dot\theta_k^{\text{new}}=\sum_\alpha O^*_{\alpha,k}r_\alpha+\underbrace{\sum_\alpha O^*_{\alpha,k}O_{\alpha,k}}_{S_{kk}}\,\dot\theta_k^{\text{cur}} .$$

### 3.4 Solve for the increment

With $\dot\theta_k^{\text{new}}=\dot\theta_k^{\text{cur}}+\delta_k$, the $S_{kk}\dot\theta_k^{\text{cur}}$ terms cancel:

$$\boxed{\;S_{kk}\,\delta_k = O_k^\dagger\, r\;} \tag{4}$$

Only the $p_k\times p_k$ block $S_{kk}$ is inverted.

### 3.5 Update the residual incrementally

Only block $k$'s contribution to $r$ has changed:

$$r\;\leftarrow\; r-O_k\,\delta_k . \tag{5}$$

This costs $O(N_sp_k)$. Recomputing $\sum_{i\neq k}O_{\alpha,i}\dot\theta_i$ from scratch would cost $O(N_sN_p)$ per block.

### 3.6 Fixed point = full TDVP

At a fixed point all $\delta_k=0$. By Eq. (4) this means $O_k^\dagger r=0$ for every $k$:

$$\sum_\alpha O^*_{\alpha,k}\Big(e_\alpha-\sum_i O_{\alpha,i}\dot\theta_i\Big)=F_k-\sum_iS_{ki}\dot\theta_i=0\quad\forall k
\;\Longleftrightarrow\; S\dot\theta=F .$$

The cross terms $S_{ki}$ were accounted for throughout, but only through the vector $r$, never as matrices. For $S$ positive definite (or $S+\lambda I$, Sec. 6), block Gauss–Seidel is guaranteed to converge for any partition and any block ordering.

## 4. Algorithm (one RK stage; $\theta$ and samples frozen)

```python
# O: (N_s, N_p) centred log-derivatives, or computed blockwise via jvp/vjp
# e: (N_s,)     centred local energies (with time-evolution phase)
# blocks: list of index arrays P_k, a disjoint partition of range(N_p)
# mu: per-block damping (Sec. 6), max_sweeps, tol

chol = [cholesky(O[:, P].conj().T @ O[:, P] + mu * I) for P in blocks]  # once per stage

thetadot = thetadot_prev                 # warm start (or zeros)
r = e - O @ thetadot                     # computed once: O(N_s N_p)

best = inf
for sweep in range(max_sweeps):
    for k in random_permutation(K):
        Ok = O[:, blocks[k]]
        delta = cho_solve(chol[k], Ok.conj().T @ r)    # Eq. (4), damped
        thetadot[blocks[k]] += delta
        r -= Ok @ delta                                # Eq. (5)
    res_hold = norm(e_hold - O_hold @ thetadot)**2 / norm(e_hold)**2
    if res_hold > best - tol:            # holdout residual stopped improving
        break
    best = res_hold
```

Then pass `thetadot` to the time integrator.

## 5. Cost and memory

| | SR | MinSR | Block Gauss–Seidel |
|---|---|---|---|
| Setup | $N_sN_p^2+N_p^3$ | $N_s^2N_p+N_s^3$ | $N_sN_pp+N_pp^2$ ($K$ Cholesky factorizations) |
| Per sweep | — | — | $N_sN_p$ |
| Matrix memory | $N_p^2$ | $N_s^2$ | $N_pp$ |

- $O$ itself ($N_s\times N_p$) can be kept, or each block can be recomputed on the fly with jvp/vjp, sample chunk by sample chunk.
- The scheme beats SR whenever the number of sweeps is much smaller than $N_p/p$. Warm starting and early stopping usually make this the case.
- Ill-conditioned, strongly coupled directions converge slowly. That is the price of never inverting $S$, and also the mechanism behind the implicit regularization in Sec. 6.

## 6. Regularization

There are two ways to regularize a block update. They differ in where the bias ends up.

### 6.1 Penalizing the value (plain Tikhonov)

$$\delta_k=\arg\min_\delta\|O_k\delta-r\|^2+\lambda\|\dot\theta_k+\delta\|^2
\;\Rightarrow\;
(S_{kk}+\lambda I)\,\delta_k = O_k^\dagger r-\lambda\dot\theta_k .$$

The fixed point satisfies $O_k^\dagger r=\lambda\dot\theta_k$ for all $k$, i.e. $(S+\lambda I)\dot\theta=F$. This carries exactly the same bias as plain TDVP with shift $\lambda$; cycling removes none of it. The same global $\lambda$ must be used in every block, otherwise the fixed point changes.

### 6.2 Penalizing the step (proximal / Levenberg–Marquardt)

$$\delta_k=\arg\min_\delta\|O_k\delta-r\|^2+\mu\|\delta\|^2
\;\Rightarrow\;
(S_{kk}+\mu I)\,\delta_k = O_k^\dagger r .$$

At a fixed point $\delta_k=0 \Rightarrow O_k^\dagger r=0$: the **unregularized** normal equations. The damping $\mu$ changes only the convergence rate, not the limit. Consequences:

- Each block can be damped heavily, with any SPD per-block damping ($\mu_k$, $\mu\,\mathrm{diag}\,S_{kk}$, …), without changing the converged solution.
- The regularization moves into the **number of sweeps** $n$.

For $K=1$, $n$ proximal iterations from zero are iterated Tikhonov. In the eigenbasis of $S$ (eigenvalues $\sigma^2$), the filter factors applied to $F_\sigma/\sigma^2$ are

$$\text{Tikhonov: } \frac{\sigma^2}{\sigma^2+\lambda},
\qquad
\text{iterated Tikhonov: } 1-\Big(\frac{\mu}{\sigma^2+\mu}\Big)^n .$$

This follows from the error recursion $\epsilon_{n+1} = \frac{\mu}{\sigma^2+\mu}\,\epsilon_n$.

- **Noise-dominated directions** ($\sigma^2\ll\mu$): filter $\approx n\sigma^2/\mu$, the same suppression as Tikhonov with $\lambda\approx\mu/n$.
- **Well-determined directions** ($\sigma^2\gg\mu$): the bias is $O\big((\mu/\sigma^2)^n\big)$ instead of $O(\lambda/\sigma^2)$.

For equal suppression of noisy directions, well-resolved directions are much less biased. For $K>1$ the filter is not diagonal in the eigenbasis of $S$, but the picture holds qualitatively. Directions coupling many blocks converge more slowly and are therefore regularized more, and these are typically the redundant ones.

### 6.3 Stopping criterion

With proximal damping, full convergence returns the unregularized TDVP solution, including its fit to Monte Carlo noise. The number of sweeps must therefore be chosen by **generalization**, not by convergence:

- Split the samples into a training set (used in the sweeps) and a holdout set.
- The training residual $\|r\|^2/\|e\|^2$, the TDVP $r^2$, decreases monotonically and is free to compute.
- The holdout residual $\|e_\text{hold}-O_\text{hold}\dot\theta\|^2/\|e_\text{hold}\|^2$ measures whether $\dot\theta$ generalizes. Stop when it stops improving.

### 6.4 Null space

With step-only damping, components of $\dot\theta$ in $\ker S$ are never corrected, so a warm start carries them along. They do not change $\dot\psi$ at this instant, but the kernel of the QGT rotates in time, and such drift can become physical later. Two remedies:

- Warm start only the range component of the previous $\dot\theta$.
- Add a tiny global value-penalty $\lambda_0$ (Sec. 6.1) purely to pin the null space.

## 7. Practical notes

- **Partition.** Re-randomize the partition each time step, or at least the block order each sweep. Layer-wise blocks capture intra-layer correlations exactly, while random blocks retain the dropout-like benefit of randomized sparse updates. Keep $p_k\lesssim N_s/10$ so that each $S_{kk}$ is well estimated.
- **Warm start.** $\dot\theta$ varies smoothly between RK stages and time steps, so starting from the previous solution typically needs only 1–3 sweeps.
- **RK stages.** Each stage has its own $\theta$ and samples, so each stage runs its own solve. The partition can stay fixed within one RK step.
- **Real parameters with complex $\psi$.** Apply the same algorithm to the real-stacked system $\big[\mathrm{Re}\,O;\ \mathrm{Im}\,O\big]$, $\big[\mathrm{Re}\,e;\ \mathrm{Im}\,e\big]$.
- **Energy conservation.** For holomorphic networks with complex parameters, any parameter subset spans a complex-linear tangent subspace, so even a single partial update conserves $E$ in continuous time.
- **Parallelism.**
  - Updating all blocks simultaneously (block Jacobi) is **not** guaranteed to converge; correlated blocks overshoot by up to a factor $K$.
  - Damped Jacobi with step $\omega\sim1/K$ is safe but slow.
  - Better: use the $K$ independent block solves as a block-diagonal preconditioner inside CG (block-Jacobi-preconditioned CG). The block solves are parallel and CG handles the coupling.
  - Note that the matvecs $O_k^\dagger r$ and $O_k\delta$ are already parallel over samples.

## 8. Numerical check

Random complex system with $N_s=400$, $N_p=120$, $K=6$ random blocks and no regularization. Relative error of $\dot\theta$ against a direct solve of $S\dot\theta=F$:

| Sweeps | 1 | 5 | 10 | 30 |
|---|---|---|---|---|
| Rel. error | $4.5\times10^{-1}$ | $2.6\times10^{-2}$ | $1.2\times10^{-3}$ | $1.3\times10^{-8}$ |

A real QGT is far worse conditioned than a random matrix, so expect slower convergence along its small-eigenvalue directions. That slowness is what early stopping exploits.

## 9. Related work

- J. Berman, B. Peherstorfer, *Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks*, NeurIPS 2023 ([arXiv:2310.04867](https://arxiv.org/abs/2310.04867)). One random parameter subset per time step, no iteration.
- Z. Chen, J. McCarran, E. Vizcaino, M. Soljačić, D. Luo, *TENG: Time-Evolving Natural Gradient for Solving PDEs with Deep Neural Nets Toward Machine Precision*, ICML 2024 ([arXiv:2404.10771](https://arxiv.org/abs/2404.10771)). Fixed-target projection per time step, with several Gauss–Newton iterations each on a random parameter subset.
- *Adaptive quantum dynamics with the time-dependent variational Monte Carlo method* ([arXiv:2506.08575](https://arxiv.org/abs/2506.08575)). tVMC restricted to an importance-selected active parameter subset.
- A. Sinibaldi, C. Giuliani, G. Carleo, F. Vicentini, *Unbiasing time-dependent Variational Monte Carlo by projected quantum evolution*, Quantum 7, 1131 (2023). p-tVMC, the quantum analogue of the fixed-target approach.