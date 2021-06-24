from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym

# TODO put this in seperate repo
# TODO test to make sure looping constructs are working like I think, ie python loop only happens once on first trace and all refs are correct on subsequent runs

def print_time(t):
    days=int(t//86400);hours=int((t-days*86400)//3600);mins=int((t-days*86400-hours*3600)//60);secs=int((t-days*86400-hours*3600-mins*60))
    return "{:4d}:{:02d}:{:02d}:{:02d}".format(days,hours,mins,secs)

def replace_infnan(inputs, replace):
    isinfnan = tf.math.logical_or(tf.math.is_nan(inputs), tf.math.is_inf(inputs))
    return tf.where(isinfnan, replace, inputs)

def discretize(inputs, min, max):
    inputs = tf.math.round(inputs)
    inputs = tf.clip_by_value(inputs, min, max)
    return inputs


class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, groups, eps=None, axis=-1, name=None):
        super(EvoNormS0, self).__init__(name=name)
        self.groups, self.axis = groups, axis
        if eps is None: eps = tf.experimental.numpy.finfo(self.compute_dtype).eps
        self.eps = tf.identity(tf.constant(eps, dtype=self.compute_dtype))

    def build(self, input_shape):
        inlen = len(input_shape)
        shape = [1] * inlen
        shape[self.axis] = input_shape[self.axis]
        self.gamma = self.add_weight(name="gamma", shape=shape, initializer=tf.keras.initializers.Ones())
        self.beta = self.add_weight(name="beta", shape=shape, initializer=tf.keras.initializers.Zeros())
        self.v1 = self.add_weight(name="v1", shape=shape, initializer=tf.keras.initializers.Ones())

        groups = min(input_shape[self.axis], self.groups)
        group_shape = input_shape.as_list()
        group_shape[self.axis] = input_shape[self.axis] // groups
        group_shape.insert(self.axis, groups)
        self.group_shape = tf.Variable(group_shape, trainable=False, name='group_shape')

        std_shape = list(range(1, inlen+self.axis))
        std_shape.append(inlen)
        self.std_shape = tf.identity(std_shape)

    @tf.function
    def call(self, inputs, training=True):
        input_shape = tf.shape(inputs)
        self.group_shape[0].assign(input_shape[0]) # use same learned parameters with different batch size
        grouped_inputs = tf.reshape(inputs, self.group_shape)
        _, var = tf.nn.moments(grouped_inputs, self.std_shape, keepdims=True)
        std = tf.sqrt(var + self.eps)
        std = tf.broadcast_to(std, self.group_shape)
        group_std = tf.reshape(std, input_shape)

        return (inputs * tf.math.sigmoid(self.v1 * inputs)) / group_std * self.gamma + self.beta



class Deterministic(tfp.layers.DistributionLambda):
    def __init__(self, event_shape=(), **kwargs):
        kwargs.pop('make_distribution_fn', None) # for get_config serializing
        params_shape = tf.identity(list(event_shape))
        super(Deterministic, self).__init__(lambda input: Deterministic.new(input, params_shape), **kwargs)
        self._event_shape = event_shape
    @staticmethod
    def new(params, params_shape):
        # print("tracing -> Deterministic new")
        output_shape = tf.concat([tf.shape(params)[:-1], params_shape], axis=0)
        params = tf.reshape(params, output_shape)
        dist = tfp.distributions.Deterministic(loc=params)
        return dist
    @staticmethod
    def params_size(event_shape=(), name=None):
        params_size = np.prod(event_shape).item()
        return params_size

class Categorical(tfp.layers.DistributionLambda):
    def __init__(self, num_components, event_shape=(), dtype_cat=tf.int32, **kwargs):
        params_shape = list(event_shape)+[num_components]
        reinterpreted_batch_ndims = len(event_shape)
        kwargs.pop('make_distribution_fn', None) # for get_config serializing
        params_shape, reinterpreted_batch_ndims = tf.identity(params_shape), tf.identity(reinterpreted_batch_ndims)
        super(Categorical, self).__init__(lambda input: Categorical.new(input, params_shape, reinterpreted_batch_ndims, dtype_cat), **kwargs)
        self._num_components, self._event_shape = num_components, event_shape
    @staticmethod
    def new(params, params_shape, reinterpreted_batch_ndims, dtype_cat=tf.int32):
        # print("tracing -> Categorical new")
        output_shape = tf.concat([tf.shape(params)[:-1], params_shape], axis=0)
        params = tf.reshape(params, output_shape)
        dist = tfp.distributions.Categorical(logits=params, dtype=dtype_cat)
        dist = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        return dist
    @staticmethod
    def params_size(num_components, event_shape=(), name=None):
        event_size = np.prod(event_shape).item()
        params_size = event_size * num_components
        return params_size

class CategoricalRP(tfp.layers.DistributionLambda): # reparametertized
    def __init__(self, event_shape=(), temperature=1e-5, **kwargs):
        compute_dtype = tf.keras.backend.floatx()
        num_components, event_shape = event_shape[-1], event_shape[:-1]
        params_shape = list(event_shape)+[num_components]
        reinterpreted_batch_ndims = len(event_shape)
        kwargs.pop('make_distribution_fn', None) # for get_config serializing
        params_shape, reinterpreted_batch_ndims, temperature = tf.identity(params_shape), tf.identity(reinterpreted_batch_ndims), tf.identity(tf.constant(temperature, compute_dtype))
        super(CategoricalRP, self).__init__(lambda input: CategoricalRP.new(input, params_shape, reinterpreted_batch_ndims, temperature), **kwargs)
        self._num_components, self._event_shape = num_components, event_shape
    @staticmethod
    def new(params, params_shape, reinterpreted_batch_ndims, temperature=1e-5):
        # print("tracing -> CategoricalRP new")
        output_shape = tf.concat([tf.shape(params)[:-1], params_shape], axis=0)
        params = tf.reshape(params, output_shape)
        dist = tfp.distributions.ExpRelaxedOneHotCategorical(temperature=temperature, logits=params)
        # dist = tfp.distributions.RelaxedOneHotCategorical(temperature=temperature, logits=params)
        # dist = tfp.distributions.RelaxedBernoulli(temperature=temperature, logits=params)
        dist = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        return dist
    @staticmethod
    def params_size(event_shape=(), name=None):
        num_components, event_shape = event_shape[-1], event_shape[:-1]
        event_size = np.prod(event_shape).item()
        params_size = event_size * num_components
        return params_size



class MixtureLogistic(tfp.layers.DistributionLambda):
    def __init__(self, num_components, event_shape=(), **kwargs):
        compute_dtype = tf.keras.backend.floatx()
        eps = tf.experimental.numpy.finfo(compute_dtype).eps
        maxroot = tf.math.sqrt(tf.dtypes.as_dtype(compute_dtype).max)

        params_shape = [num_components]+list(event_shape)
        reinterpreted_batch_ndims = len(event_shape)
        
        kwargs.pop('make_distribution_fn', None) # for get_config serializing
        num_components, params_shape, reinterpreted_batch_ndims, eps, maxroot = tf.identity(num_components), tf.identity(params_shape), tf.identity(reinterpreted_batch_ndims), tf.identity(tf.constant(eps, compute_dtype)), tf.identity(tf.constant(maxroot, compute_dtype))
        super(MixtureLogistic, self).__init__(lambda input: MixtureLogistic.new(input, num_components, params_shape, reinterpreted_batch_ndims, eps, maxroot), **kwargs)
        self._num_components, self._event_shape = num_components, event_shape

    @staticmethod # this doesn't change anything, just keeps the variables seperate
    def new(params, num_components, params_shape, reinterpreted_batch_ndims, eps, maxroot):
        # print("tracing -> MixtureLogistic new")
        mixture_params = params[..., :num_components]

        components_params = params[..., num_components:]
        loc_params, scale_params = tf.split(components_params, 2, axis=-1)

        output_shape = tf.concat([tf.shape(params)[:-1], params_shape], axis=0)
        loc_params = tf.reshape(loc_params, output_shape)
        
        scale_params = tf.math.abs(scale_params)
        # scale_params = tf.clip_by_value(scale_params, eps, maxroot)
        scale_params = tfp.math.clip_by_value_preserve_gradient(scale_params, eps, maxroot)
        scale_params = tf.reshape(scale_params, output_shape)

        dist = tfp.distributions.MixtureSameFamily(
                mixture_distribution = tfp.distributions.Categorical(
                    logits=mixture_params
                ),
                components_distribution = tfp.distributions.Independent(
                    # tfp.distributions.Normal(
                    tfp.distributions.Logistic(
                        loc=loc_params,
                        scale=scale_params
                    ),
                    reinterpreted_batch_ndims=reinterpreted_batch_ndims
                ),
            # reparameterize=True, # better spread of loc and scale params, rep net works better
        )
        return dist
    @staticmethod
    def params_size(num_components, event_shape=(), name=None):
        event_size = np.prod(event_shape).item()
        params_size = num_components + event_size * num_components * 2
        return params_size



from tensorflow.python.ops import special_math_ops
class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, num_heads, latent_size, memory_size, **kwargs):
        compute_dtype = tf.keras.backend.floatx()
        key_dim = int(latent_size/num_heads)
        super(MultiHeadAttention, self).__init__(tf.identity(num_heads), tf.identity(key_dim), **kwargs)

        mem_zero = tf.constant(np.full((1, memory_size, latent_size), 0), compute_dtype)
        self._mem_size, self._mem_zero = tf.identity(memory_size), tf.identity(mem_zero)
    
    def build(self, input_shape):
        self._mem_idx = tf.Variable(self._mem_size, trainable=False, name='mem_idx')
        self._memory = tf.Variable(self._mem_zero, trainable=False, name='memory')

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        attention_scores = special_math_ops.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_output = special_math_ops.einsum(self._combine_equation, attention_scores, value)
        return attention_output, attention_scores

    def call(self, query, attention_mask=None, training=None):
        if not self._built_from_signature: self._build_from_signature(query=query, value=query, key=None)

        time_size = tf.shape(query)[1]

        drop_off = tf.roll(self._memory, shift=-time_size, axis=1)
        self._memory.assign(drop_off)
        self._memory[:,-time_size:].assign(query)

        if self._mem_idx > 0:
            mem_idx_next = self._mem_idx - time_size
            if mem_idx_next < 0: mem_idx_next = 0
            self._mem_idx.assign(mem_idx_next)

        value = self._memory[:,self._mem_idx:]
        seq_size = tf.shape(value)[1]

        query = self._query_dense(query)
        key = self._key_dense(value)
        value = self._value_dense(value)

        if training: attention_mask = tf.linalg.band_part(tf.ones((time_size,seq_size)), -1, seq_size - time_size)

        attention_output, attention_scores = self._compute_attention(query, key, value, attention_mask)
        
        scores = tf.math.reduce_sum(attention_scores, axis=(1,2))[0]
        scores = tf.argsort(scores, axis=-1, direction='ASCENDING', stable=True)
        scores = tf.gather(self._memory[:,self._mem_idx:], scores, axis=1)
        self._memory[:,self._mem_idx:].assign(scores)

        attention_output = self._output_dense(attention_output)
        return attention_output

    def reset_states(self):
        self._mem_idx.assign(self._mem_size)
        self._memory.assign(self._mem_zero)




def gym_get_space_zero(space):
    if isinstance(space, gym.spaces.Discrete): zero = np.asarray(0, space.dtype)
    elif isinstance(space, gym.spaces.Box): zero = np.zeros(space.shape, space.dtype)
    elif isinstance(space, gym.spaces.Tuple):
        zero = [None]*len(space.spaces)
        for i,s in enumerate(space.spaces): zero[i] = gym_get_space_zero(s)
        zero = tuple(zero)
    elif isinstance(space, gym.spaces.Dict):
        zero = OrderedDict()
        for k,s in space.spaces.items(): zero[k] = gym_get_space_zero(s)
    return zero

# TODO add different kinds of net_type? 0 = Dense, 1 = 2 layer Dense, 2 = Conv2D, etc
def gym_get_spec(space, compute_dtype='float64', force_cont=False):
    if isinstance(space, gym.spaces.Discrete):
        dtype = tf.dtypes.as_dtype(space.dtype)
        dtype_out = compute_dtype if force_cont else 'int32'
        dtype_out = tf.dtypes.as_dtype(dtype_out)
        spec = [{'net_type':0, 'dtype':dtype, 'dtype_out':dtype_out, 'min':tf.constant(0,dtype_out), 'max':tf.constant(space.n-1,dtype_out), 'is_discrete':True, 'num_components':space.n, 'event_shape':(1,)}]
        zero, zero_out = [tf.constant([[0]], dtype)], [tf.constant([[0]], dtype_out)]
    elif isinstance(space, gym.spaces.Box):
        dtype = tf.dtypes.as_dtype(space.dtype)
        dtype_out = tf.dtypes.as_dtype(compute_dtype)
        spec = [{'net_type':0, 'dtype':dtype, 'dtype_out':dtype_out, 'min':tf.constant(space.low,dtype_out), 'max':tf.constant(space.high,dtype_out), 'is_discrete':False, 'num_components':np.prod(space.shape).item(), 'event_shape':space.shape}]
        zero, zero_out = [tf.zeros([1]+list(space.shape), dtype)], [tf.zeros([1]+list(space.shape), dtype_out)]
    elif isinstance(space, (gym.spaces.Tuple, gym.spaces.Dict)):
        spec, zero, zero_out = [], [], []
        loop = space.spaces.items() if isinstance(space, gym.spaces.Dict) else enumerate(space.spaces)
        for k,s in loop:
            spec_sub, zero_sub, zero_out_sub = gym_get_spec(s, compute_dtype, force_cont)
            spec += spec_sub; zero += zero_sub; zero_out += zero_out_sub
    return spec, zero, zero_out


# TODO test tf.nest.flatten(data)
def gym_space_to_feat(data, space):
    feat = []
    if isinstance(data, tuple):
        for i,v in enumerate(data): feat += gym_space_to_feat(v, space[i])
    elif isinstance(data, dict):
        for k,v in data.items(): feat += gym_space_to_feat(v, space[k])
    elif isinstance(data, np.ndarray): feat = [np.expand_dims(data,0)]
    else: feat = [np.asarray([[data]], space.dtype)]
    return feat

# TODO test tf.nest.pack_sequence_as(out, space)
def gym_out_to_space(out, space, idx):
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.Box)):
        data = out[idx[0]]
        if isinstance(space, gym.spaces.Discrete): data = data.item() # numpy.int64 is coming in here in graph mode
        idx[0] += 1
    elif isinstance(space, gym.spaces.Tuple):
        data = [None]*len(space.spaces)
        for i,s in enumerate(space.spaces): data[i] = gym_out_to_space(out, s, idx)
        data = tuple(data)
    elif isinstance(space, gym.spaces.Dict):
        data = OrderedDict()
        for k,s in space.spaces.items(): data[k] = gym_out_to_space(out, s, idx)
    return data


def gym_struc_to_feat(data):
    if data.dtype.names is None:
        if len(data.shape) == 1: data = np.expand_dims(data,-1)
        feat = [data]
    else:
        feat = []
        for k in data.dtype.names: feat += gym_struc_to_feat(data[k])
    return feat

def gym_out_to_struc(out, dtype):
    for i in range(len(out)): out[i] = np.frombuffer(out[i], dtype=np.uint8)
    out = np.concatenate(out)
    out = np.frombuffer(out, dtype=dtype)
    return out


def gym_space_to_bytes(data, space):
    byts = []
    if isinstance(data, tuple):
        for i,v in enumerate(data): byts += gym_space_to_bytes(v, space[i])
    elif isinstance(data, dict):
        for k,v in data.items(): byts += gym_space_to_bytes(v, space[k])
    else:
        if not isinstance(data, np.ndarray): data = np.asarray(data, space.dtype)
        byts = [np.frombuffer(data, dtype=np.uint8)]
    return byts

def gym_bytes_to_space(byts, space, idxs, idx):
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.Box)):
        data = byts[idxs[idx[0]]:idxs[idx[0]+1]]
        if space.dtype != np.uint8: data = np.frombuffer(data, dtype=space.dtype)
        if isinstance(space, gym.spaces.Box) and len(space.shape) > 1: data = np.reshape(data, space.shape)
        if isinstance(space, gym.spaces.Discrete): data = data.item()
        idx[0] += 1
    elif isinstance(space, gym.spaces.Tuple):
        data = [None]*len(space.spaces)
        for i,s in enumerate(space.spaces): data[i] = gym_bytes_to_space(byts, s, idxs, idx)
        data = tuple(data)
    elif isinstance(space, gym.spaces.Dict):
        data = OrderedDict()
        for k,s in space.spaces.items(): data[k] = gym_bytes_to_space(byts, s, idxs, idx)
    return data
