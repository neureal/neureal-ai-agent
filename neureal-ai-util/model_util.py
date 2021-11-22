from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym, numba

# TODO put this in seperate repo
# TODO test to make sure looping constructs are working like I think, ie python loop only happens once on first trace and all refs are correct on subsequent runs

def print_time(t):
    days=int(t//86400);hours=int((t-days*86400)//3600);mins=int((t-days*86400-hours*3600)//60);secs=int((t-days*86400-hours*3600-mins*60))
    return "{:4d}:{:02d}:{:02d}:{:02d}".format(days,hours,mins,secs)

def replace_infnan(inputs, replace):
    isinfnan = tf.math.logical_or(tf.math.is_nan(inputs), tf.math.is_inf(inputs))
    return tf.where(isinfnan, replace, inputs)

# TODO tf.keras.layers.Discretization ?
def discretize(inputs, spec, force_cont):
    if force_cont and spec['is_discrete']: inputs = tf.math.round(inputs)
    if spec['dtype'] == tf.uint8 or spec['dtype'] == tf.int32 or spec['dtype'] == tf.int64: inputs = tf.math.round(inputs)
    inputs = tf.clip_by_value(inputs, spec['min'], spec['max'])
    inputs = tf.cast(inputs, spec['dtype'])
    # inputs = tf.dtypes.saturate_cast(inputs, spec['dtype'])
    inputs = tf.squeeze(inputs)
    return inputs

@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True)
def ewma(arr_in, window):
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=numba.float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma

@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True)
def ewma_ih(arr_in, window): # infinite history, faster
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=numba.float64)
    alpha = 2 / float(window + 1)
    ewma[0] = arr_in[0]
    for i in range(1, n): ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
    return ewma



class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, groups, eps=None, axis=-1, name=None):
        super(EvoNormS0, self).__init__(name=name)
        self._groups, self._axis = groups, axis
        if eps is None: eps = tf.experimental.numpy.finfo(self.compute_dtype).eps
        self._eps = tf.identity(tf.constant(eps, dtype=self.compute_dtype))

    def build(self, input_shape):
        inlen = len(input_shape)
        shape = [1] * inlen
        shape[self._axis] = input_shape[self._axis]
        self._gamma = self.add_weight(shape=shape, initializer=tf.keras.initializers.Ones(), name='gamma')
        self._beta = self.add_weight(shape=shape, initializer=tf.keras.initializers.Zeros(), name='beta')
        self._v1 = self.add_weight(shape=shape, initializer=tf.keras.initializers.Ones(), name='v1')

        groups = min(input_shape[self._axis], self._groups)
        group_shape = input_shape[self._axis:].as_list()
        group_shape[self._axis] = input_shape[self._axis] // groups
        group_shape.insert(self._axis, groups)
        self._group_shape = tf.identity(group_shape)

        std_shape = list(range(1, inlen + self._axis))
        std_shape.append(inlen)
        self._std_shape = tf.identity(std_shape)

    @tf.function
    def call(self, inputs, training=True):
        input_shape = tf.shape(inputs)
        group_shape = tf.concat([input_shape[:self._axis], self._group_shape], axis=0)
        grouped_inputs = tf.reshape(inputs, group_shape)
        _, var = tf.nn.moments(grouped_inputs, self._std_shape, keepdims=True)
        std = tf.sqrt(var + self._eps)
        std = tf.broadcast_to(std, group_shape)
        group_std = tf.reshape(std, input_shape)

        return (inputs * tf.math.sigmoid(self._v1 * inputs)) / group_std * self._gamma + self._beta



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


class MixtureSameFamily(tfp.distributions.MixtureSameFamily):
    def _entropy(self):
        # entropy = self.components_distribution.entropy() * self.mixture_distribution.probs_parameter()
        # entropy = tf.reduce_sum(entropy, axis=1) # entropy1
        # entropy = tf.concat([self.components_distribution.entropy(), tf.expand_dims(self.mixture_distribution.entropy(), axis=1)], axis=1)
        # entropy = tf.reduce_mean(entropy, axis=1) # entropy2
        entropy = self.sample(256)
        entropy = -self.log_prob(entropy)
        entropy = tf.reduce_mean(entropy, axis=0) # entropy3
        return entropy

class Logistic(tfp.distributions.Logistic):
    def _log_prob(self, x):
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        z = (x - loc) / (scale)
        return -z - 2. * tf.math.softplus(-z) - tf.math.log1p(scale)
        # return -z - 2. * tf.math.softplus(-z) - tf.math.log(scale)

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

        batch_size = tf.shape(params)[:-1]
        output_shape = tf.concat([batch_size, params_shape], axis=0)
        loc_params = tf.reshape(loc_params, output_shape)
        
        scale_params = tf.math.abs(scale_params)
        # scale_params = tfp.math.clip_by_value_preserve_gradient(scale_params, eps, maxroot)
        scale_params = tf.clip_by_value(scale_params, eps, maxroot)
        scale_params = tf.reshape(scale_params, output_shape)

        dist_mixture = tfp.distributions.Categorical(logits=mixture_params)
        # dist_component = tfp.distributions.Normal(loc=loc_params, scale=scale_params)
        # dist_component = tfp.distributions.Logistic(loc=loc_params, scale=scale_params)
        dist_component = Logistic(loc=loc_params, scale=scale_params)
        dist_components = tfp.distributions.Independent(dist_component, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        dist = MixtureSameFamily(mixture_distribution=dist_mixture, components_distribution=dist_components)
        # dist = MixtureSameFamily(mixture_distribution=dist_mixture, components_distribution=dist_components, reparameterize=True) # better spread of loc and scale params, rep net works better, can have bugs

        return dist
    @staticmethod
    def params_size(num_components, event_shape=(), name=None):
        event_size = np.prod(event_shape).item()
        params_size = num_components + event_size * num_components * 2
        return params_size



from tensorflow.python.ops import special_math_ops
class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, latent_size, num_heads=1, memory_size=None, sort_memory=False, norm=False, hidden_size=None, evo=None, residual=True, use_bias=False, cross_type=None, num_latents=None, channels=None, init_zero=None, **kwargs): # cross_type: 1 = input, 2 = output
        # key_dim = int(channels/num_heads) if cross_type == 2 else int(latent_size/num_heads)
        key_dim = int(latent_size/num_heads)
        super(MultiHeadAttention, self).__init__(tf.identity(num_heads), tf.identity(key_dim), use_bias=use_bias, **kwargs)
        self._mem_size, self._sort_memory, self._norm, self._residual, self._cross_type = memory_size, sort_memory, norm, residual, cross_type

        if memory_size is not None:
            mem_channels = latent_size if cross_type != 1 else channels
            mem_zero = tf.constant(np.full((1, memory_size, mem_channels), 0), tf.keras.backend.floatx())
            self._mem_zero = tf.identity(mem_zero)
            if sort_memory:
                mem_score_zero = tf.constant(np.full((1, memory_size), 0), tf.keras.backend.floatx())
                self._mem_score_zero = tf.identity(mem_score_zero)

        if norm:
            # float_eps = tf.experimental.numpy.finfo(tf.keras.backend.floatx()).eps
            # self._layer_norm_key = tf.keras.layers.LayerNormalization(epsilon=float_eps, center=True, scale=True, name='norm_key')
            # self._layer_norm_value = tf.keras.layers.LayerNormalization(epsilon=float_eps, center=True, scale=True, name='norm_value')
            self._layer_dense_key_in = tf.keras.layers.Dense(hidden_size, activation=EvoNormS0(evo), use_bias=False, name='dense_key_in')
            self._layer_dense_value_in = tf.keras.layers.Dense(hidden_size, activation=EvoNormS0(evo), use_bias=False, name='dense_value_in')

        # query_scale = 1.0 / tf.math.sqrt(tf.cast(self._key_dim, dtype=tf.keras.backend.floatx()))
        # self._query_scale = tf.identity(query_scale)

        if cross_type: # CrossAttention
            if init_zero is None:
                # init_zero = tf.random.normal((1, num_latents, latent_size), mean=0.0, stddev=0.02, dtype=tf.keras.backend.floatx())
                # init_zero = tf.clip_by_value(init_zero, -2.0, 2.0)
                if cross_type == 1: # input, batch = different latents
                    init_zero = np.linspace(-1.0, 1.0, num_latents) # -np.e, np.e
                    init_zero = np.expand_dims(init_zero, axis=-1)
                    init_zero = np.repeat(init_zero, latent_size, axis=-1)
                if cross_type == 2: # output, batch = actual batch
                    init_zero = np.linspace(-1.0, 1.0, channels) # -np.e, np.e
                    init_zero = np.expand_dims(init_zero, axis=0)
                    init_zero = np.repeat(init_zero, num_latents, axis=0)
                init_zero = np.expand_dims(init_zero, axis=0)
            init_zero = tf.constant(init_zero, dtype=tf.keras.backend.floatx())
            self._init_zero = tf.identity(init_zero)

    def build(self, input_shape):
        if self._mem_size is not None:
            self._mem_idx, self._memory = tf.Variable(self._mem_size, trainable=False, name='mem_idx'), tf.Variable(self._mem_zero, trainable=False, name='memory')
            self._mem_idx_img, self._memory_img = tf.Variable(self._mem_size, trainable=False, name='mem_idx_img'), tf.Variable(self._mem_zero, trainable=False, name='memory_img')
            if self._sort_memory:
                self._mem_score = tf.Variable(self._mem_score_zero, trainable=False, name='mem_score')
                self._mem_score_img = tf.Variable(self._mem_score_zero, trainable=False, name='mem_score_img')
        if self._cross_type: self._init_latent = tf.Variable(self._init_zero, trainable=False, name='init_latent')
        if self._residual: self._residual_amt = tf.Variable(0.0, dtype=self.compute_dtype, trainable=True, name='residual') # ReZero

    def _compute_attention(self, query, key, value, attention_mask=None):
        # query = tf.math.multiply(query, self._query_scale)
        attn_scores = special_math_ops.einsum(self._dot_product_equation, key, query)
        attn_scores = self._masked_softmax(attn_scores, attention_mask) # TODO can I replace softmax here with somthing more log likelihood related? (ie continuous attn)
        attn_output = special_math_ops.einsum(self._combine_equation, attn_scores, value)
        return attn_output, attn_scores

    def call(self, value, attention_mask=None, auto_mask=None, store_memory=True, use_img=False, num_latents=None):
        if self._cross_type:
            # value[0](batch) = time dim, value[1:-2] = space/feat dim, value[-1] = channel dim
            value = tf.reshape(value, (1, -1, tf.shape(value)[-1]))
            query = self._init_latent if num_latents is None else self._init_latent[:,:num_latents]
        else: query = value
        time_size = tf.shape(value)[1]
        if not self._built_from_signature: self._build_from_signature(query=query, value=value, key=None)

        if self._mem_size is not None:
            if use_img: mem_idx, memory = self._mem_idx_img, self._memory_img
            else: mem_idx, memory = self._mem_idx, self._memory
            if self._sort_memory: mem_score = self._mem_score_img if use_img else self._mem_score

        if self._mem_size is not None and store_memory:
            drop_off = tf.roll(memory, shift=-time_size, axis=1)
            memory.assign(drop_off)
            memory[:,-time_size:].assign(value)

            if self._sort_memory:
                drop_off = tf.roll(mem_score, shift=-time_size, axis=1)
                mem_score.assign(drop_off)
                zero = self._mem_score_zero[:,-time_size:]
                mem_score[:,-time_size:].assign(zero)

            if mem_idx > 0:
                mem_idx_next = mem_idx - time_size
                if mem_idx_next < 0: mem_idx_next = 0
                mem_idx.assign(mem_idx_next)

        if self._mem_size is not None:
            # value_mem = memory[:,mem_idx:] # gradients not always working with this
            if not store_memory and use_img and mem_idx != self._mem_idx:
                now_idx = mem_idx - self._mem_idx
                value_mem = tf.concat([memory[:,mem_idx:now_idx-time_size], value, memory[:,now_idx:]], axis=1)
            else: value_mem = tf.concat([memory[:,mem_idx:-time_size], value], axis=1)
        else: value_mem = value

        # TODO loop through value or query if too big for memory
        query_ = self._query_dense(query)
        if self._norm:
            key = self._layer_dense_key_in(value_mem)
            key = self._key_dense(key)
        else: key = self._key_dense(value_mem)
        # if self._norm: key = self._layer_norm_key(key)
        if self._norm:
            value = self._layer_dense_value_in(value_mem)
            value = self._value_dense(value)
        else: value = self._value_dense(value_mem)
        # if self._norm: value = self._layer_norm_value(value)

        seq_size = tf.shape(query)[1]
        if auto_mask: attention_mask = tf.linalg.band_part(tf.ones((time_size,seq_size)), -1, seq_size - time_size)

        attn_output, attn_scores = self._compute_attention(query_, key, value, attention_mask)

        if self._mem_size is not None and store_memory and self._sort_memory:
            # scores = tf.math.reduce_sum(attn_scores, axis=(1,2))[0]
            # scores = tf.argsort(scores, axis=-1, direction='ASCENDING', stable=True)
            # scores = tf.gather(memory[:,mem_idx:], scores, axis=1)
            # memory[:,mem_idx:].assign(scores)

            scores = tf.math.reduce_mean(attn_scores, axis=(1,2))[0] # heads
            # norm = tf.cast(tf.shape(scores)[0], dtype=self.compute_dtype)
            # scores = tf.math.multiply(scores, norm)
            scores = tf.math.add(mem_score[:,mem_idx:], scores)
            mem_score[:,mem_idx:].assign(scores)

            scores = tf.argsort(scores[0], axis=-1, direction='ASCENDING', stable=True)

            mem_sorted = tf.gather(memory[:,mem_idx:], scores, axis=1)
            memory[:,mem_idx:].assign(mem_sorted)

            scores_sorted = tf.gather(mem_score[:,mem_idx:], scores, axis=1)
            mem_score[:,mem_idx:].assign(scores_sorted)

        attn_output = self._output_dense(attn_output)
        if self._residual: attn_output = query + attn_output * self._residual_amt # ReZero
        if self._cross_type: attn_output = tf.squeeze(attn_output, axis=0)
        return attn_output

    def reset_states(self, use_img=False):
        if self._mem_size is not None:
            if use_img:
                self._mem_idx_img.assign(self._mem_idx)
                self._memory_img.assign(self._memory)
                if self._sort_memory: self._mem_score_img.assign(self._mem_score)
            else:
                self._mem_idx.assign(self._mem_size)
                self._memory.assign(self._mem_zero)
                if self._sort_memory: self._mem_score.assign(self._mem_score_zero)



class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, latent_size, evo=None, residual=True, **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        if evo is None: self._layer_dense = tf.keras.layers.Dense(hidden_size, activation=tf.keras.activations.gelu, use_bias=False, name='dense')
        else: self._layer_dense = tf.keras.layers.Dense(hidden_size, activation=EvoNormS0(evo), use_bias=False, name='dense')
        self._layer_dense_logits = tf.keras.layers.Dense(latent_size, name='dense_logits')
        self._residual = residual
        
    def build(self, input_shape):
        if self._residual: self._residual_amt = tf.Variable(0.0, dtype=self.compute_dtype, trainable=True, name='residual') # ReZero

    def call(self, input):
        out = self._layer_dense(input)
        out = self._layer_dense_logits(out)
        if self._residual: out = input + out * self._residual_amt # Rezero
        return out




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
        spec = [{'net_type':0, 'dtype':dtype, 'dtype_out':dtype_out, 'min':tf.constant(0,dtype_out), 'max':tf.constant(space.n-1,dtype_out), 'is_discrete':True, 'num_components':space.n, 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1))}]
        zero, zero_out = [tf.constant([[0]], dtype)], [tf.constant([[0]], dtype_out)]
    elif isinstance(space, gym.spaces.Box):
        dtype = tf.dtypes.as_dtype(space.dtype)
        dtype_out = tf.dtypes.as_dtype(compute_dtype)
        spec = [{'net_type':0, 'dtype':dtype, 'dtype_out':dtype_out, 'min':tf.constant(space.low,dtype_out), 'max':tf.constant(space.high,dtype_out), 'is_discrete':False, 'num_components':int(np.prod(space.shape).item()), 'event_shape':space.shape, 'event_size':int(np.prod(space.shape[:-1]).item()), 'channels':space.shape[-1], 'step_shape':tf.TensorShape([1]+list(space.shape))}]
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
        if isinstance(space, gym.spaces.Box): data = np.reshape(data, space.shape)
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
        if isinstance(space, gym.spaces.Box): data = np.reshape(data, space.shape)
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
