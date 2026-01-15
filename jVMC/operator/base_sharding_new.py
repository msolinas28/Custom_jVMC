from abc import ABC, abstractmethod
import jax.numpy as jnp
import inspect
import jax

from jVMC.vqs import NQS
from jVMC.sharding_config import ShardedMethod, DEVICE_SPEC

op_dtype = jnp.complex128

def _has_kwargs(fun):
    sig = inspect.signature(fun)
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

def _mul_prefactor(a, b):
    if callable(a) and callable(b):
        return lambda **kw: a(**kw) * b(**kw)
    if callable(a):
        return lambda **kw: a(**kw) * b
    if callable(b):
        return lambda **kw: a * b(**kw)
    return lambda **kw: a * b

class OperatorString(list):
    """
    A list of operators to be applied sequentially, with an associated scale factor.
    """
    def __init__(self, operators):
        super().__init__(operators)
        self._scale = lambda **kw: 1
        self._diagonal = False 

    @property
    def scale(self):
        return self._scale
    
    @property
    def diagonal(self):
        return self._diagonal

class Operator(ABC):
    def __init__(self, ldim, idx, diag, fermionic):
        self._ldim = ldim
        self._idx = idx
        self._diag = diag
        self._fermionic = fermionic
        self._is_compiled = False
        self._scale = 1

    @property
    def ldim(self):
        return self._ldim

    @property
    def idx(self):
        return self._idx
    
    @property
    def fermionic(self):
        return self._fermionic
    
    @property
    def diag(self):
        return self._diag
    
    @property
    @abstractmethod
    def mat_els(self):
        pass
    
    @property
    @abstractmethod
    def map(self):
        pass
    
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            # TODO: Since I don't know the total dim of the Hilbert space this is not doable
            raise NotImplementedError 
        elif isinstance(other, Operator):
            return CompositeOperator(self, other, 'sum')
        else:
            raise NotImplemented
        
    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            # TODO: Same as previous todo
            raise NotImplementedError
        else:
            raise NotImplemented
        
    def __neg__(self):
        return ScaledOperator(self, -1)
    
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            # TODO: Same as previous todo
            raise NotImplementedError
        elif isinstance(other, Operator):
            return CompositeOperator(self, -other, 'sum')
        else:
            raise NotImplemented
        
    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            # TODO: Same as previous todo
            raise NotImplementedError
        else:
            raise NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) or callable(other):
            return ScaledOperator(self, other)
        elif isinstance(other, Operator):
            return CompositeOperator(self, other, 'mul')
        else:
            raise NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)) or callable(other):
            return ScaledOperator(self, other)
        else:
            raise NotImplemented
        
    def get_list_of_strings(self):
        """
        Flatten the operator tree into a list of operator strings.
        Each operator string is an OperatorString (list subclass) of leaf operators 
        to be applied sequentially, with a scale attribute.
        
        Returns: List of OperatorString objects
        """
        strings_stack = []
        op_stack = [(self, False)]

        while op_stack:
            node, visited = op_stack.pop()

            if not visited:
                op_stack.append((node, True))

                if isinstance(node, CompositeOperator):
                    op_stack.append((node.O_2, False))
                    op_stack.append((node.O_1, False))

                elif isinstance(node, ScaledOperator):
                    op_stack.append((node.O, False))

            else:
                if isinstance(node, CompositeOperator):
                    right = strings_stack.pop()
                    left = strings_stack.pop()

                    if node._label == 'sum':
                        # Sum: concatenate the two lists of operator strings
                        strings_stack.append(left + right)
                    
                    elif node._label == 'mul':
                        # Product: create all combinations (distributive law)
                        out = []
                        for left_string in left:
                            for right_string in right:
                                # Combine the operators from both strings
                                combined = OperatorString(left_string + right_string)
                                # Multiply the prefactors
                                combined._scale = _mul_prefactor(left_string._scale, right_string._scale)
                                out.append(combined)
                        
                        strings_stack.append(out)

                elif isinstance(node, ScaledOperator):
                    lst = strings_stack.pop()
                    # Multiply the scale of each operator string
                    for s in lst:
                        s._scale = _mul_prefactor(s._scale, node.scalar)
                    strings_stack.append(lst)

                else:
                    # Leaf operator: wrap in an OperatorString
                    op_string = OperatorString([node])
                    strings_stack.append([op_string])        
        
        return strings_stack[0]

    def _compile(self):
        """
        Compile the operator into JAX arrays for efficient computation.
        This should be called once before using get_O_loc.
        """
        strings = self.get_list_of_strings()
        max_length = max(len(s) for s in strings)
        
        # Create identity operator for padding
        IdOp = IdentityOperator(self.ldim, 0)
        
        idxC = []
        mapC = []
        matElsC = []
        fermionicC = []
        diagonal = []
        prefactors = []
        
        for op_string in strings:
            idx_row = []
            map_row = []
            matels_row = []
            fermionic_row = []
            d = 1
            
            n = len(op_string)
            for k in range(max_length):
                k_rev = n - 1 - k
                if k_rev >= 0:
                    op = op_string[k_rev]
                    idx_row.append(op.idx)
                    map_row.append(op.map)
                    matels_row.append(op.mat_els)
                    fermionic_row.append(1.0 if op.fermionic else 0.0)
                    d *= op.diag
                else:
                    idx_row.append(IdOp.idx)
                    map_row.append(IdOp.map)
                    matels_row.append(IdOp.mat_els)
                    fermionic_row.append(0.0)
            
            idxC.append(idx_row)
            mapC.append(map_row)
            matElsC.append(matels_row)
            fermionicC.append(fermionic_row)
            prefactors.append(op_string.scale)
            diagonal.append(d)
        
        self.idxC = jnp.array(idxC, dtype=jnp.int32)
        self.mapC = jnp.array(mapC, dtype=jnp.int32)
        self.matElsC = jnp.array(matElsC)
        self.fermionicC = jnp.array(fermionicC, dtype=jnp.int32)
        self.diagC = jnp.array(diagonal, dtype=jnp.int32)
        self.prefactorsC = prefactors
        self._is_compiled = True
    
    def get_O_loc(self, s, psi: NQS, **kwargs):
        if 'logPsiS' in kwargs.keys():
            logPsiS = kwargs['logPsiS']
            kwargs.pop('logPsiS')
        else:
            logPsiS = psi(s)

        s_p, matEls = self.get_conn_elements(s, psi, **kwargs)
        logPsiS_p = psi(s_p) 

        # Automatically sharded since everything is on device
        def O_loc(ls, lsp, m):
            return jnp.sum(jnp.exp(lsp - ls[:,None]) * m, axis=1)

        return jax.jit(O_loc)(logPsiS, logPsiS_p, matEls) # TODO: Find a way to batch this 

    def get_conn_elements(self, s, psi: NQS, **kwargs):
        if not self._is_compiled:
            self._compile()

        return self._get_conn_elements_(s, psi=psi, **kwargs)

    @ShardedMethod(out_specs=(DEVICE_SPEC, DEVICE_SPEC), attr_source='psi')
    def _get_conn_elements_(self, s, **kwargs):
        return  lambda p, x, kw: self._get_conn_elements(x, kw)

    def _get_conn_elements(self, s, kwargs):
        sampleShape = s.shape
        s = s.ravel()
        dim = s.shape[0]
        mask = jnp.tril(jnp.ones((dim, dim), dtype=int), -1).T
        prefactors = jnp.array([p(**kwargs) for p in self.prefactorsC], dtype=op_dtype)

        def proccess_string(s, prefactor, idx, map, matEls, fermi):
            def apply_operator(c, x):
                carry_sample, carry_matEl = c
                idx, map, matEl, fermi = x

                fermi_sign = jnp.prod((1 - 2 * fermi) * (2 * fermi * mask[idx] + (1 - 2 * fermi)) * carry_sample + (1 - abs(carry_sample)))
                carry_matEl_new = carry_matEl * matEl[carry_sample[idx]] * fermi_sign
                carry_sample_new = carry_sample.at[idx].set(map[carry_sample[idx]])

                return (carry_sample_new, carry_matEl_new), None
            
            (s_p, matEl), _ = jax.lax.scan(apply_operator, (s, prefactor), (idx, map, matEls, fermi))

            return s_p.reshape(sampleShape), matEl
        
        return jax.vmap(proccess_string, in_axes=(None,) + (0,) * 5)(s, prefactors, self.idxC, self.mapC, self.matElsC, self.fermionicC)

class IdentityOperator(Operator):
    def __init__(self, ldim, idx):
        super().__init__(ldim, idx, True, False)

    @property
    def mat_els(self):
        return jnp.ones(self.ldim, dtype=op_dtype)
    
    @property
    def map(self):
        return jnp.arange(self.ldim, dtype=jnp.int16)

class CompositeOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator, label: str):
        if O_1.ldim != O_2.ldim:
            raise ValueError(f'The {label} is implemented only for operators with the same local dimension')
        super().__init__(O_1.ldim, None, None, None)

        self.O_1 = O_1
        self.O_2 = O_2
        self._label = label

    @property
    def mat_els(self):
        pass
    
    @property
    def map(self):
        pass

class ScaledOperator(Operator):
    def  __init__(self, O: Operator, scalar):
        if callable(scalar) and (not _has_kwargs(scalar)):
            raise ValueError('Any callable that multiplies an operator has to have **kwargs in its argument.')

        super().__init__(O.ldim, O.idx, O.diag, O.fermionic)
        self.scalar = scalar
        self.O = O

    @property
    def mat_els(self):
        pass
    
    @property
    def map(self):
        pass