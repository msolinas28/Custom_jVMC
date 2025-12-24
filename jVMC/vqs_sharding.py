import jax
jax.config.update("jax_enable_x64", True)
from jax import grad 
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.experimental.shard_map import shard_map
import flax
from flax.core.frozen_dict import freeze
import numpy as np
import flax.linen as nn
import collections
from math import isclose
from functools import partial, wraps
import warnings

import jVMC
from jVMC.util.key_gen import generate_seed
from jVMC.sharding_config import MESH, DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING
from jVMC.sharding_config import distribute, broadcast_split_key

def flat_gradient(fun, params, arg):
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: x.ravel(), gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: x.ravel(), gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_cpx_nonholo(fun, params, arg):
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gr))[0]
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gi))[0]

    return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

def flat_gradient_real(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: x.ravel(), g))[0]

    return jnp.concatenate(g)

def flat_gradient_holo(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_flatten(tree_map(lambda x: [x.ravel(), 1.j*x.ravel()], g))[0]

    return jnp.concatenate(g)

def dict_gradient(fun, params, arg):
    gr = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    gr = tree_map(lambda x: x.ravel(), gr)
    gi = grad(lambda p, y: jnp.imag(fun(p, y)))(params, arg)["params"]
    gi = tree_map(lambda x: x.ravel(), gi)

    return tree_map(lambda x,y: x + 1.j*y, gr, gi)

def dict_gradient_real(fun, params, arg):
    g = grad(lambda p, y: jnp.real(fun(p, y)))(params, arg)["params"]
    g = tree_map(lambda x: x.ravel(), g)

    return g

def create_batches(configs, b):
    append = b * ((configs.shape[0] + b - 1) // b) - configs.shape[0]
    pads = [(0, append), ] + [(0, 0)] * (len(configs.shape) - 1)

    return jnp.pad(configs, pads).reshape((-1, b) + configs.shape[1:])

class ShardedMethod:
    """Decorator to automatically create sharded versions of methods."""
    def __init__(self, use_batch=True, vmap_in_axes=None, in_specs=None, out_specs=DEVICE_SPEC, returns_dict=False):
        """
        Args:
            use_batch: Whether to create batched version
            vmap_in_axes: Additional vmap axis (will be added to (None, 0))
            in_specs: Additional input sharding specs (will be added to (REPLICATED_SPEC, DEVICE_SPEC))
            out_specs: Output sharding spec
            returns_dict: If True, uses tree_map for dict reshaping in batched version
        """
        self.use_batch = use_batch
        self.vmap_in_axes = (None, 0)
        if vmap_in_axes is not None:
            self.vmap_in_axes += vmap_in_axes
        self.in_specs = (REPLICATED_SPEC, DEVICE_SPEC)
        if in_specs is not None:
            self.in_specs += in_specs
        self.out_specs = out_specs
        self.returns_dict = returns_dict
        
    def __call__(self, method):
        """Wraps the method to create sharded versions."""
        @wraps(method)
        def wrapper(instance, s, *args, **kwargs):
            if not hasattr(instance, '_sharded_cache'):
                instance._sharded_cache = {}
            
            method_name = method.__name__
            if method_name not in instance._sharded_cache:
                base_fn = method(instance, None, *args, **kwargs)  # Call with None to get the lambda
                instance._sharded_cache[method_name] = self._create_sharded_versions(instance, base_fn)
            
            cache = instance._sharded_cache[method_name]
            return self._handle_sharding_cases(instance, cache, s, *args, **kwargs)        
        return wrapper
    
    def _create_sharded_versions(self, instance, base_fn):
        """Create all sharded versions of the method."""
        single_fn = lambda p, x, *args, **kwargs: base_fn(instance, p, x, *args, **kwargs)
        vmap_fn = jax.vmap(single_fn, in_axes=self.vmap_in_axes)
        jsh_fn = jax.jit(shard_map(vmap_fn, MESH, self.in_specs, self.out_specs))
        
        if self.use_batch:
            batched_fn = partial(self._batched_wrapper, instance=instance, vmap_fn=vmap_fn)
            batched_jsh_fn = jax.jit(shard_map(batched_fn, MESH, self.in_specs, self.out_specs))
        else:
            batched_jsh_fn = None
        
        return {
            'single': single_fn,
            'vmap': vmap_fn,
            'jsh': jsh_fn,
            'batched_jsh': batched_jsh_fn
        }
    
    def _handle_sharding_cases(self, instance, cache, s, *args, **kwargs):
        """Helper function to handle different sharding cases."""
        # Case: single sample
        if s.shape == instance.sampleShape:
            return cache['single'](instance.parameters, s, *args, **kwargs)
        
        num_devices = MESH.size
        total_samples = s.shape[0]
        
        # Case: fewer samples than devices -> fall back to vmap
        if total_samples < num_devices:
            return cache['vmap'](instance.parameters, s, *args, **kwargs)
        
        # Case: not divisible by num_devices -> pad
        original_size = total_samples
        if total_samples % num_devices != 0:
            pad_size = num_devices - (total_samples % num_devices)
            s = jnp.concatenate([s, jnp.repeat(s[-1:], pad_size, axis=0)], axis=0)
            total_samples = s.shape[0]
        
        # Calculate samples per device
        samples_per_device = total_samples // num_devices
        
        # Case: batch_size is None or samples_per_device <= batch_size
        if instance.batchSize is None or samples_per_device <= instance.batchSize:
            result = cache['jsh'](instance.parameters, s, *args, **kwargs)
        else:
            result = cache['batched_jsh'](instance.parameters, s, *args, **kwargs)
        
        # Trim padding
        if original_size != total_samples:
            result = jax.tree_util.tree_map(lambda x: x[:original_size], result)
        
        return result
    
    def _batched_wrapper(self, parameters, s, *args, instance, vmap_fn, **kwargs):
        """Wrapper for batched computation."""
        sb = create_batches(s, instance.batchSize)
        print('ciao')
        
        def scan_fun(c, x):
            return c, vmap_fn(parameters, x, *args, **kwargs)
        result = jax.lax.scan(scan_fun, None, jnp.array(sb))[1]
        
        if self.returns_dict:
            result = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), result)
            return jax.tree_util.tree_map(lambda x: x[:s.shape[0]], result)
        else:
            return result.reshape((-1,))[:s.shape[0]]
        
class NQS:
    """
        Initializes NQS class.
        
        This class can operate in two modi:
            #. Single-network ansatz
                Quantum state of the form :math:`\\psi_\\theta(s)\\equiv\\exp(r_\\theta(s))`, \
                where the network :math:`r_\\theta` is
                a) holomorphic, i.e., parametrized by complex valued parameters :math:`\\vartheta`.
                b) non-holomorphic, i.e., parametrized by real valued parameters :math:`\\theta`.
            #. Two-network ansatz
                Quantum state of the form 
                :math:`\\psi_\\theta(s)\\equiv\\exp(r_{\\theta_r}(s)+i\\varphi_{\\theta_\\phi}(s))` \
                with an amplitude network :math:`r_{\\theta_{r}}` and a phase network \
                :math:`\\varphi_{\\theta_\\phi}` \
                parametrized by real valued parameters :math:`\\theta_r,\\theta_\\phi`.
        Args:       
            * ``net``: Variational network or tuple of networks.
                A network has to be registered as pytree node and provide \
                a ``__call__`` function for evaluation. \
                If a tuple of two networks is given, the first is used for the logarithmic \
                amplitude and the second for the phase of the wave function coefficient.
            * ``logarithmic``: Boolean variable indicating, whether the ANN returns logarithmic \
                (:math:`\\log\\psi_\\theta(s)`) or plain (:math:`\\psi_\\theta(s)`) wave function coefficients.
            * ``batchSize``: Batch size for batched network evaluation. Choice \
                of this parameter impacts performance: with too small values performance \
                is limited by memory access overheads, too large values can lead \
                to "out of memory" issues.
            * ``seed``: Seed for the PRNG to initialize the network parameters.
            * ``orbit``: Orbit which defining the symmetry operations (instance of ``util.symmetries.LatticeSymmetry``). \
                If this argument is given, the wave function is symmetrized to be invariant under symmetry operations.
            * ``avgFun``: Reduction operation for the symmetrization.
        """

    def __init__(self, net: nn.Module, sampleShape, logarithmic=True, batchSize: None | int = None, 
                 seed: None | int = None, orbit=None, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Exp):
        if isinstance(net, collections.abc.Iterable):
            for n in net:
                if not isinstance(n, nn.Module):
                    raise ValueError("The argument 'net' has to be an instance of flax.nn.Module.")
            net = jVMC.nets.two_nets_wrapper.TwoNets(net)
        else:
            if not isinstance(net, nn.Module):
                raise ValueError("The argument 'net' has to be an instance of flax.nn.Module.")
        self._isGenerator = False
        if "sample" in dir(net):
            if callable(net.sample):
                self._isGenerator = True
        if orbit is not None:
            net = jVMC.nets.sym_wrapper.SymNet(net=net, orbit=orbit, avgFun=avgFun)
        self._net = net

        if isinstance(sampleShape, tuple):
            self._sampleShape = sampleShape
        else:
            self._sampleShape = (sampleShape,)
        
        self._batchSize = batchSize
        self._logarithmic = logarithmic
        
        self.init_net(seed)
        if self.logarithmic:
            self.apply_fun = self.net.apply
        else:
            def apply_fun(parameters, s, method=None):
                return jnp.log(self.net.apply(parameters, s, method))
            self.apply_fun = apply_fun

        self._append_gradients_dict_single = lambda x, y: tree_map(lambda a, b: jnp.concatenate((a, 1.j * b)), x, y)
        self._append_gradients_dict = lambda x, y: tree_map(lambda a, b: jnp.concatenate((a[:, :], 1.j * b[:, :]), axis=1), x, y)
        self._append_gradients_dict_jsh = jax.jit(shard_map(self._append_gradients_dict, MESH, (DEVICE_SPEC, DEVICE_SPEC), DEVICE_SPEC))
        
        self._sharded_cache = {}

    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        if hasattr(value, "shape"):
            value = self._param_unflatten(value)
        if 'params' not in value.keys():
            value = {'params': value}
        if isinstance(self.parameters, flax.core.frozen_dict.FrozenDict):
            value = freeze(value)
        self._parameters = value

    @property
    def params(self):
        return self.parameters['params']
    
    @params.setter
    def params(self, value):
        self.parameters = value

    @property
    def batchSize(self):
        return self._batchSize
        
    @property
    def net(self):
        return self._net
    
    @property
    def sampleShape(self):
        return self._sampleShape
    
    @property
    def logarithmic(self):
        return self._logarithmic

    @property
    def is_generator(self):
        return self._isGenerator
    
    @property
    def holomorphic(self):
        return self._holomorphic
    
    @property
    def realParams(self):
        return self._realParams
    
    @property
    def flat_gradient_function(self):
        return self._flat_gradient_function
    
    @property
    def dict_gradient_function(self):
        return self._dict_gradient_function
    
    @property
    def paramShapes(self):
        return self._paramShapes
    
    @property
    def numParameters(self):
        return self._numParameters
        
    def init_net(self, seed: int | None):
        dummy_sample = jnp.ones(self.sampleShape)
        if seed == None:
            seed = generate_seed()
        self._parameters = self.net.init(jax.random.PRNGKey(seed), dummy_sample)
        self._realParams = False
        self._holomorphic = False
        
        dtypes = [a.dtype for a in tree_flatten(self.parameters)[0]]
        if not all(d == dtypes[0] for d in dtypes):
            raise Exception("Network uses different parameter data types. This is not supported.")
        if dtypes[0] == np.single or dtypes[0] == np.double:
            self._realParams = True

        # check Cauchy-Riemann condition to test for holomorphicity
        def make_flat(t):
            return jnp.concatenate([p.ravel() for p in tree_flatten(t)[0]])
        
        grads_r = make_flat(jax.grad(lambda a, b: jnp.real(self.net.apply(a,b)))(self.parameters, dummy_sample)["params"])
        grads_i = make_flat(jax.grad(lambda a, b: jnp.imag(self.net.apply(a,b)))(self.parameters, dummy_sample)["params"])
        if isclose(jnp.linalg.norm(grads_r - 1.j * grads_i) / grads_r.shape[0], 0.0, abs_tol=1e-14):
            self._holomorphic = True
            self._flat_gradient_function = flat_gradient_holo
            self._dict_gradient_function = dict_gradient_real
        else:
            if self._realParams:
                self._flat_gradient_function = flat_gradient
                self._dict_gradient_function = dict_gradient
            else:
                self._flat_gradient_function = flat_gradient_cpx_nonholo
                self._dict_gradient_function = dict_gradient_real

        self._paramShapes = [(p.size, p.shape) for p in tree_flatten(self.parameters["params"])[0]]
        self._netTreeDef = jax.tree_util.tree_structure(self.parameters["params"])
        self._numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.parameters["params"])[0]]))
    
    @ShardedMethod()
    def __call__(self, s):
        """
        Evaluate variational wave function.
        
        Compute the logarithmic wave function coefficients :math:`\\ln\\psi(s)` for \
        computational configurations :math:`s`.
        
        Args:
            * ``s``: Array of computational basis states.
        Returns:
            Logarithmic wave function coefficients :math:`\\ln\\psi(s)`.
        
        :meta public:
        """ 
        return lambda self, p, x: self.apply_fun(p, x)

    @ShardedMethod(returns_dict=True)
    def gradients(self, s):
        """
        Compute gradients of logarithmic wave function.
        
        Compute gradient of the logarithmic wave function coefficients, \
        :math:`\\nabla\\ln\\psi(s)`, for computational configurations :math:`s`.
        
        Args:
            * ``s``: Array of computational basis states.
        Returns:
            A vector containing derivatives :math:`\\partial_{\\theta_k}\\ln\\psi(s)` \
            with respect to each variational parameter :math:`\\theta_k` for each \
            input configuration :math:`s`.
        """
        return lambda self, p, x: self.flat_gradient_function(self.net.apply, p, x)
    
    def gradients_dict(self, s):
        result = self._gradients_dict(s)

        if self.holomorphic:
            if s.shape == self.sampleShape:
                return self._append_gradients_dict_single(result, result)
            else:
                return self._append_gradients_dict_jsh(result, result)
        return result
    
    @ShardedMethod(returns_dict=True)
    def _gradients_dict(self, s):
        return lambda self, p, x: self.dict_gradient_function(self.net.apply, p, x)

    def grad_dict_to_vec_map(self):
        PTreeShape = []
        start = 0
        P = jnp.arange(2 * self.numParameters)
        for s in self.paramShapes:
            # Here we need to add the treatment for the complex non-holomorphic case
            if self.holomorphic:
                PTreeShape.append((P[start:start + 2 * s[0]]))
                start += 2 * s[0]
            else:
                PTreeShape.append(P[start:start + s[0]])
                start += s[0]
        
        return tree_unflatten(self._netTreeDef, PTreeShape)

    def get_sampler_net(self):
        """
        Get real part of NQS and current parameters

        This function returns a function that evaluates the real part of the NQS,
        :math:`\\text{Re}(\\log\\psi(s))`, and the current parameters.

        Returns:
            Real part of the NQS and current parameters
        """
        if "eval_real" in dir(self.net):
            if callable(self.net.eval_real):
                return lambda p, x: jnp.real(self.apply_fun(p, x, method=self.net.eval_real)), self.parameters
        else:
            return lambda p, x: jnp.real(self.apply_fun(p, x)), self.parameters

    def sample(self, numSamples, key=None, parameters=None):
        if self._isGenerator:
            params = self.parameters
            if parameters is not None:
                params = parameters

            numSamples = distribute(numSamples, 'samples') # TODO: after this check if batchsize divides numSaples per device
            if key is None:
                key = jax.random.PRNGKey(generate_seed())
            elif len(key.shape) > 1:
                key = key[0]
            keys = jax.device_put(broadcast_split_key(key, numSamples), DEVICE_SHARDING)
    
            numSamplesStr = str(numSamples)
            # check whether _get_samples is already compiled for given number of samples
            if not numSamplesStr in self._sample_jsh:
                self._sample_jsh[numSamplesStr] = jax.jit(
                    shard_map(
                        lambda p, n, k: self.net.apply(p, n, k, method=self.net.sample), 
                        mesh=MESH,
                        in_specs=(REPLICATED_SPEC,) * 2 + (DEVICE_SPEC,),
                        out_specs=(DEVICE_SPEC,)
                    )
                )

            return self._sample_jsh[numSamplesStr](params, numSamples, keys)

        return None
 
    def update_parameters(self, deltaP):
        """
        Update variational parameters.
        
        Sets new values of all variational parameters by adding given values.
        If parameters are not initialized, parameters are set to ``deltaP``.
        
        Args:
            * ``deltaP``: Values to be added to variational parameters.
        """
        self.parameters = jax.tree_util.tree_map(jax.lax.add, self.params, self._param_unflatten(deltaP))

    def set_parameters(self, P):
        """
        Set variational parameters.
        
        Sets new values of all variational parameters.
        
        Args:
            * ``P``: New values of variational parameters.
        """
        warnings.warn(
            "set_parameters() is deprecated and will be removed in a future release; "
            "assign to `self.parameters` or 'self.params' directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Update model parameters
        if isinstance(P, flax.core.frozen_dict.FrozenDict):
            self.params = P
        else:
            self.params = self._param_unflatten(P)

    # TODO: change name to make it more explicit that it returns the params in array form
    def get_parameters(self):
        """
        Get variational parameters.
        
        Returns:
            Array holding current values of all variational parameters.
        """
        if not self._realParams:
            return jnp.concatenate([jnp.concatenate([p.ravel().real, p.ravel().imag]) for p in tree_flatten(self.params)[0]])
        return jnp.concatenate([p.ravel() for p in tree_flatten(self.params)[0]])

    def _param_unflatten(self, P):
        """
        Reshape parameter array update according to net tree structure
        """
        if isinstance(P, dict):
            if 'params' in P.keys():
                return P['params']
            else:
                return P
        PTreeShape = []
        start = 0
        for s in self.paramShapes:
            if not self._realParams:
                PTreeShape.append((P[start:start + s[0]] + 1.j * P[start + s[0]:start + 2 * s[0]]).reshape(s[1]))
                start += 2 * s[0]
            else:
                PTreeShape.append(P[start:start + s[0]].reshape(s[1]))
                start += s[0]
        
        return tree_unflatten(self._netTreeDef, PTreeShape)