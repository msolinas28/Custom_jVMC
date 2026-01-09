from abc import ABC, abstractmethod
import jax.numpy as jnp
import inspect

from jVMC.vqs import NQS

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
    return a * b

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
    
    @property #TODO: fix
    def _n_strings(self):
        if len(self.mat_els.shape) == 1:
            return 1
        else:
            return self.mat_els.shape[0] 
    
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
    
    # def _flatten(self):
    #     strings_stack = []
    #     op_stack = [(self, False)]

    #     while op_stack:
    #         node, visited = op_stack.pop()

    #         if not visited:
    #             op_stack.append((node, True))

    #             if isinstance(node, CompositeOperator):
    #                 op_stack.append((node.O_2, False))
    #                 op_stack.append((node.O_1, False))

    #             elif isinstance(node, ScaledOperator):
    #                 op_stack.append((node.O, False))

    #         else:
    #             if isinstance(node, CompositeOperator):
    #                 right = strings_stack.pop()
    #                 left = strings_stack.pop()

    #                 if node._label == 'sum':
    #                     strings_stack.append(left + right)
                    
    #                 elif node._label == 'mul':
    #                     right = strings_stack.pop()
    #                     left = strings_stack.pop()
    #                     out = []
    #                     for a in left:
    #                         for b in right:
    #                             out.append([a, b])
    #                     strings_stack.append(out)

    #             elif isinstance(node, ScaledOperator):
    #                 lst = strings_stack.pop()
    #                 for s in lst:
    #                     s._scale = _mul_prefactor(s._scale, node.scalar)
    #                 strings_stack.append(lst)

    #             else:
    #                 strings_stack.append(node)

    #     return strings_stack

        # # merge diagonal-only strings
        # merged = []
        # diag_acc = None

        # for s in strings_stack[0]:
        #     if s.diag:
        #         if diag_acc is None:
        #             diag_acc = s
        #         else:
        #             diag_acc.prefactor = _add_prefactor(
        #                 diag_acc.prefactor, s.prefactor
        #             )
        #     else:
        #         merged.append(s)

        # if diag_acc is not None:
        #     merged.insert(0, diag_acc)

        # self._strings = merged
        # self._is_flattened = True

    def _get_conn():
        # to shard with decorator
        pass

    def _get_O_loc():
        # put here the ratio times the matmul
        pass

    def get_O_loc(self, s, psi: NQS, **kwargs):
        pass 
        # if not self._is_compiled:
        #     self._compile()

        # if 'logPsiS' in kwargs.keys():
        #     logPsiS = kwargs['logPsiS']
        #     kwargs.pop('logPsiS')
        # else:
        #     logPsiS = psi(s)

        # mat_elements, sp = self._get_conn() # the shape of theoutput must be standardized otherwise it will continue to recompile in the next call
        # logPsiSp = psi(sp) 

        # return self._get_O_loc(mat_elements, logPsiS, logPsiSp)

    def _get_operator_strings(self):
        """
        Flatten the operator tree into a list of operator strings.
        Each operator string is a list of leaf operators to be applied sequentially.
        
        Returns: List of operator strings, where each string is [op1, op2, ...] with a _scale attribute
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
                        strings_stack.append(left + right)
                    
                    elif node._label == 'mul':
                        out = []
                        for left_string in left:
                            for right_string in right:
                                op_string = OperatorString(left_string.operators + right_string.operators)
                                op_string._scale = _mul_prefactor(left_string._scale, right_string._scale)
                                out.append(op_string)
                        
                        strings_stack.append(out)

                elif isinstance(node, ScaledOperator):
                    lst = strings_stack.pop()
                    for s in lst:
                        s._scale = _mul_prefactor(s._scale, node.scalar)
                    strings_stack.append(lst)

                else:
                    op_string = OperatorString(node)
                    strings_stack.append([op_string])

        return strings_stack[0]

class OperatorString:
    """
    Represents a single operator string: a sequence of operators to be applied.
    Example: [B, C] means apply B then C
    """
    def __init__(self, operators):
        self.operators = operators if isinstance(operators, list) else [operators]
        self._scale = 1
    
    @property
    def idx(self):
        """Concatenated indices from all operators in the string."""
        result = []
        for op in self.operators:
            if isinstance(op.idx, list):
                result.extend(op.idx)
            else:
                result.append(op.idx)
        return result
    
    @property
    def map(self):
        """Concatenated maps from all operators in the string."""
        result = []
        for op in self.operators:
            if isinstance(op.map, list):
                result.extend(op.map)
            else:
                result.append(op.map)
        return result
    
    @property
    def mat_els(self):
        """Concatenated matrix elements from all operators in the string."""
        result = []
        for op in self.operators:
            if isinstance(op.mat_els, list):
                result.extend(op.mat_els)
            else:
                result.append(op.mat_els)
        return result
    
    @property
    def fermionic(self):
        """Concatenated fermionic flags from all operators in the string."""
        result = []
        for op in self.operators:
            if isinstance(op.fermionic, list):
                result.extend(op.fermionic)
            else:
                result.append(op.fermionic)
        return result
    
    @property
    def diag(self):
        """Check if entire string is diagonal."""
        for op in self.operators:
            if isinstance(op.diag, list):
                if not all(op.diag):
                    return False
            else:
                if not op.diag:
                    return False
        return True
    
    @property
    def ldim(self):
        """Local dimension."""
        return self.operators[0].ldim
    
    @property
    def prefactor(self):
        """The accumulated scale factor."""
        return self._scale

        





        
class IdentityOperator(Operator):
    def __init__(self, ldim, idx):
        super().__init__(ldim, idx, True, False)

    @property
    def mat_els(self):
        return jnp.ones(self.ldim)
    
    @property
    def map(self):
        return jnp.arange(self.ldim)

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