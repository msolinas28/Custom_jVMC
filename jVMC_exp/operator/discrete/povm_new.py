"""
SKETCH -- not wired up / not tested. Working notes for how a branch_free-style
POVM operator could look.

Two responsibilities that the old povm.py conflated into `POVM`/`POVMOperator`
are split here into two classes:

* `PovmBasis`   -- the measurement basis (M, T_inv) and the library of named
                   channels (dense matrices), e.g. "X", "dephasing", "ZZ".
                   Pure linear algebra, not an Operator, nothing to compile.
                   Port `get_M`/`get_paulis`/`matrix_to_povm`/`get_dissipators`/
                   `get_unitaries`/`get_observables` from the old file into this
                   more or less unchanged.

* `Operator`    -- a composable operator term, built via `basis.op(name, *sites)`
                   and combined with `+` / scalar `*`, mirroring branch_free's
                   `Operator`/`CompositeOperator`/`ScaledOperator` tree. The
                   difference from branch_free: a POVM leaf is a dense
                   `(4**k, 4**k)` block tagged with the k sites it acts on,
                   not a `map`/`mat_els` pair -- because POVM channels mix all
                   4 outcomes, not just one. So instead of chaining leaves
                   with `lax.scan` (single deterministic branch per step),
                   compiling just collects a flat list of independent blocks,
                   and evaluation enumerates all 4**max_k outputs per block
                   (same trick the old `_get_s_primes` used).

Inherits `discrete.base.Operator`, not `branch_free.Operator`: the
single-branch assumption is local to branch_free.py, the compiled
get_O_loc/get_conn_elements/to_sparse machinery in discrete/base.py is
generic and reusable as-is.

Open questions (see chat):
  * Do we keep the old compile()'s optimization of merging terms that share
    sites into one block, or accept the simpler "one block per added term"
    version below and revisit only if branch count becomes a real cost?
  * `Operator * Operator` is only implemented for disjoint site sets (tensor
    product of the two blocks). Same-site products don't have an obvious
    generic meaning for dense POVM generators -- left unsupported for now.
  * Prefactors that are callables (time-dependent strengths) can't be vmapped
    over directly; branch_free handles this with `jax.lax.switch` over a
    *Python list* of closures, indexed by a vmapped integer id. We need the
    same trick here (sketched, not fleshed out, in `_get_conn_elements`).
"""
from __future__ import annotations
import itertools
import jax
import jax.numpy as jnp

from jVMC_exp.operator.discrete.base import Operator as BaseOperator
from jVMC_exp import global_defs

opDtype = global_defs.DT_OPERATORS_REAL


class PovmBasis:
    """Measurement basis + library of named channels.

    TODO: port get_M / get_paulis / matrix_to_povm / get_dissipators /
    get_unitaries / get_observables from the old povm.py here, fixing
    imports/dtypes only -- none of that logic needs to change.
    """

    def __init__(self, theta=0, phi=0, name="SIC"):
        self.theta, self.phi, self.name = theta, phi, name
        self.M = None       # = get_M(theta, phi, name)
        self.T_inv = None   # = inv(einsum('aij,bji->ab', M, M))
        self.channels = {}  # name -> dense (4**k, 4**k) matrix

        # self.channels.update(get_unitaries(self.M, self.T_inv))
        # self.channels.update(get_dissipators(self.M, self.T_inv))
        self.observables = {}  # = get_observables(self.M, self.T_inv)

    def add_channel(self, name, mat):
        if name in self.channels:
            raise ValueError(f"There already exists a channel named '{name}'.")
        k = round(jnp.log(mat.shape[0]) / jnp.log(4))
        if mat.shape != (4**k, 4**k):
            raise ValueError(f"Channel matrix must be square with side 4**k, got shape {mat.shape}.")
        self.channels[name] = jnp.asarray(mat, dtype=opDtype)

    def op(self, name, *sites) -> "Operator":
        """Build a composable Operator leaf: `name` acting jointly on `sites`."""
        mat = self.channels[name]
        k = len(sites)
        if mat.shape != (4**k, 4**k):
            raise ValueError(
                f"Channel '{name}' has arity {round(jnp.log(mat.shape[0]) / jnp.log(4))}, "
                f"but got {k} site(s): {sites}."
            )
        return Operator(self, tuple(sites), mat)


class Operator(BaseOperator):
    """One POVM operator term (leaf, sum-node, or scaled-node).

    Leaves carry (basis, sites, mat); CompositeOperator/ScaledOperator carry
    child references instead and are only ever walked structurally by
    `_get_terms`, never compiled/evaluated directly (same convention as
    branch_free's CompositeOperator/ScaledOperator).
    """

    def __init__(self, basis: "PovmBasis | None", sites: tuple | None, mat):
        super().__init__(ldim=4)  # POVM local dim is always 4, unlike branch_free
        self.basis = basis
        self.sites = sites
        self.mat = mat

    # __add__ / __sub__ / __mul__ / __truediv__ etc. are inherited from
    # discrete.base.Operator and already dispatch to _create_composite /
    # _create_scaled below -- nothing to override here.

    @classmethod
    def _create_composite(cls, O_1: "Operator", O_2: "Operator", label: str) -> "Operator":
        if label == "mul":
            return _tensor_disjoint(O_1, O_2)
        return CompositeOperator(O_1, O_2)

    @classmethod
    def _create_scaled(cls, O: "Operator", scalar) -> "Operator":
        return ScaledOperator(O, scalar)

    def _get_terms(self) -> list["_Term"]:
        """
        Flatten the +/scale tree into a flat list of `_Term(sites, mat,
        prefactor)`, in the same iterative-stack style as branch_free's
        `_get_list_of_strings` (avoids recursion depth issues for long sums).
        """
        stack = []
        node_stack = [(self, 1)]  # (node, accumulated scalar prefactor so far)

        while node_stack:
            node, scale = node_stack.pop()
            if isinstance(node, CompositeOperator):
                node_stack.append((node.O_1, scale))
                node_stack.append((node.O_2, scale))
            elif isinstance(node, ScaledOperator):
                node_stack.append((node.O, _combine_scale(scale, node.scalar)))
            else:
                stack.append(_Term(node.sites, node.mat, scale))

        return stack

    def _compile(self):
        terms = self._get_terms()
        max_k = max(len(t.sites) for t in terms)
        max_site = max(max(t.sites) for t in terms)

        site_couplings = []
        mat_els = []
        prefactor_fns = []  # Python list of callables/scalars, selected via lax.switch

        for term in terms:
            sites, mat = _pad_block(term.sites, term.mat, max_k, max_site)
            site_couplings.append(sites)
            mat_els.append(mat.reshape((4,) * (2 * max_k)))
            prefactor_fns.append(term.prefactor if callable(term.prefactor) else (lambda **kw, v=term.prefactor: v))

        self._max_k = max_k
        self.site_couplings = jnp.array(site_couplings, dtype=jnp.int32)     # (num_terms, max_k)
        self.mat_els = jnp.array(mat_els, dtype=opDtype)                     # (num_terms, *[4]*2*max_k)
        self.prefactor_fns = prefactor_fns                                    # static Python list, len num_terms
        self._is_compiled = True

    def _get_conn_elements(self, s, kwargs):
        max_k = self._max_k
        num_terms = self.site_couplings.shape[0]

        # Static (compile-time) menu of all possible k-site output tuples.
        local_out = jnp.array(list(itertools.product(range(4), repeat=max_k)), dtype=jnp.int32)  # (4**max_k, max_k)

        def one_term(term_id, sites, mat):
            in_idx = tuple(s[sites])
            coeffs = mat[in_idx].reshape(-1)                       # (4**max_k,)
            prefactor = jax.lax.switch(term_id, self.prefactor_fns, **kwargs)
            coeffs = coeffs * prefactor

            s_p = jnp.tile(s, (4**max_k, 1))
            s_p = s_p.at[:, sites].set(local_out)
            return s_p, coeffs

        term_ids = jnp.arange(num_terms)
        s_p, mat_els = jax.vmap(one_term, in_axes=(0, 0, 0))(
            term_ids, self.site_couplings, self.mat_els
        )
        return s_p.reshape((-1,) + s.shape), mat_els.reshape(-1)


class CompositeOperator(Operator):
    """A `+` node."""
    def __init__(self, O_1: Operator, O_2: Operator):
        super().__init__(None, None, None)
        self.O_1 = O_1
        self.O_2 = O_2


class ScaledOperator(Operator):
    """A `* scalar_or_callable` node."""
    def __init__(self, O: Operator, scalar):
        super().__init__(None, None, None)
        self.O = O
        self.scalar = scalar


class _Term:
    """Flat (sites, dense matrix, scalar-or-callable prefactor) leaf, produced by `_get_terms`."""
    __slots__ = ("sites", "mat", "prefactor")

    def __init__(self, sites, mat, prefactor):
        self.sites = sites
        self.mat = mat
        self.prefactor = prefactor


def _combine_scale(scale, scalar):
    if callable(scale) or callable(scalar):
        raise NotImplementedError("TODO: compose two callables/scalars into one prefactor fn.")
    return scale * scalar


def _pad_block(sites, mat, max_k, max_site):
    """
    Embed a k-site block into a max_k-site block, acting as identity on the
    extra sites. Same trick as the old compile(): pick unused site labels
    (cycling through 0..max_site) and kron an I_4 in for each.

    TODO: port verbatim from old compile()'s while-loop + jnp.kron(mat, I_4).
    """
    raise NotImplementedError


def _tensor_disjoint(O_1: Operator, O_2: Operator) -> Operator:
    """
    `Operator * Operator` for two leaves on disjoint sites: build one bigger
    leaf on the union of sites via jnp.kron(O_1.mat, O_2.mat). Only makes
    sense for two genuine leaves (not sums) with non-overlapping `sites`;
    raise otherwise.

    TODO: implement; raise a clear error for overlapping sites or non-leaf
    operands (a leaf's `.sites`/`.mat` are only meaningful for leaves).
    """
    raise NotImplementedError
