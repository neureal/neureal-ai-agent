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



class Categorical(tfp.layers.DistributionLambda):
    def __init__(self, num_components, event_shape=(), dtype_cat=tf.int32, **kwargs):
        params_shape = list(event_shape)+[num_components]
        reinterpreted_batch_ndims = len(event_shape)
        kwargs.pop('make_distribution_fn', None) # for get_config serializing
        params_shape, reinterpreted_batch_ndims = tf.identity(params_shape), tf.identity(reinterpreted_batch_ndims)
        super(Categorical, self).__init__(lambda input: Categorical.new(input, params_shape, reinterpreted_batch_ndims, dtype_cat), **kwargs)
        self._num_components, self._event_shape = num_components, event_shape
    @staticmethod # this doesn't change anything, just keeps the variables seperate
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
    def __init__(self, num_components, event_shape=(), temperature=1e-5, **kwargs):
        compute_dtype = tf.keras.backend.floatx()
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
    # @staticmethod
    # def sample(dist, name=None):
    #     sample = dist.sample()
    #     sample = tf.exp(sample)
    #     sample = tf.math.argmax(sample, axis=-1)
    #     # sample = tf.math.argmax(sample, axis=-1, output_type=tf.int64) # cant calculate gradients
    #     # sample = tf.cast(sample, dtype=tf.float64)
    #     return sample
    @staticmethod
    def params_size(num_components, event_shape=(), name=None):
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
    def sample(dist, name=None):
        return dist.sample()

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




# def gym_get_spec(space):
#     is_vec, sample = True, None
#     # TODO expand the Dict of observation types (env.observation_space) to auto make input embedding networks
#     sample = obs_space.sample()
#     if isinstance(obs_space, gym.spaces.Discrete):
#         sample = tf.convert_to_tensor([[sample]], dtype=dtype)
#     elif isinstance(obs_space, gym.spaces.Box):
#         if len(obs_space.shape) > 1: is_vec = False
#         sample = tf.convert_to_tensor([sample], dtype=dtype)
#     # elif isinstance(obs_space, gym.spaces.Tuple):
#     elif isinstance(obs_space, gym.spaces.Dict):
#         sample = gym_flatten(obs_space, sample, dtype=dtype)
#         sample = tf.convert_to_tensor([sample], dtype=dtype)

#     return dtypes, spec, sample


# def gym_space_fill(spaces, fill):
#     for space in spaces:
#         s = space if isinstance(spaces, tuple) else spaces[space]
#         if isinstance(s, (dict,tuple)): gym_space_fill(s, fill)
#         else: s.fill(fill)

def gym_flatten(space, x, dtype=np.float32): # works with numpy data type objects for x
    if isinstance(space, gym.spaces.Tuple): return np.concatenate([gym_flatten(s, x_part, dtype) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, gym.spaces.Dict): return np.concatenate([gym_flatten(s, x[key], dtype) for key, s in space.spaces.items()])
    elif isinstance(space, gym.spaces.Discrete):
        onehot = np.zeros(space.n, dtype=dtype); onehot[x] = 1.0; return onehot
    else: return np.asarray(x, dtype=dtype).flatten()

def gym_action_dist(action_space, dtype=tf.float32, cat_dtype=tf.int32, num_components=16):
    params_size, is_discrete, dists = 0, True, None

    if isinstance(action_space, gym.spaces.Discrete):
        params_size = action_space.n
        if action_space.n < 2: dists = tfp.layers.IndependentBernoulli(1, sample_dtype=tf.bool)
        else: dists = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Categorical(logits=input, dtype=cat_dtype))
    elif isinstance(action_space, gym.spaces.Box):
        event_shape = list(action_space.shape)
        event_size = np.prod(event_shape).item()
        if action_space.dtype == np.uint8:
            params_size = event_size * 256
            if event_size == 1: dists = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Categorical(logits=input, dtype=cat_dtype))
            else:
                params_shape = event_shape+[256]
                dists = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Independent(
                    tfp.distributions.Categorical(logits=tf.reshape(input, tf.concat([tf.shape(input)[:-1], params_shape], axis=0)), dtype=cat_dtype), reinterpreted_batch_ndims=len(event_shape)
                ))
        elif action_space.dtype == np.float32 or action_space.dtype == np.float64:
            if len(event_shape) == 1: # arbitrary, but with big event shapes the paramater size is HUGE
                # params_size = tfp.layers.MixtureSameFamily.params_size(num_components, component_params_size=tfp.layers.MultivariateNormalTriL.params_size(event_size))
                # dists = tfp.layers.MixtureSameFamily(num_components, tfp.layers.MultivariateNormalTriL(event_size))
                params_size = tfp.layers.MixtureLogistic.params_size(num_components, event_shape)
                dists = tfp.layers.MixtureLogistic(num_components, event_shape)
            else:
                params_size = tfp.layers.MixtureLogistic.params_size(num_components, event_shape)
                dists = tfp.layers.MixtureLogistic(num_components, event_shape)
            # self.bijector = tfp.bijectors.Sigmoid(low=-1.0, high=1.0)
            is_discrete = False
    # TODO expand to handle Tuple+Dict, make so discrete and continuous can be output at the same time
    # event_shape/event_size = action_size['action_dist_pair'] + action_size['action_dist_percent']
    elif isinstance(action_space, gym.spaces.Tuple):
        dists = [None]*len(action_space.spaces)
        for i,space in enumerate(action_space.spaces):
            dists[i] = gym_action_dist(space, dtype=dtype, cat_dtype=cat_dtype, num_components=num_components)
            params_size += dists[i][0]
            if not dists[i][1]: is_discrete = False
    elif isinstance(action_space, gym.spaces.Dict):
        dists = {}
        for k,space in action_space.spaces.items():
            dists[k] = gym_action_dist(space, dtype=dtype, cat_dtype=cat_dtype, num_components=num_components)
            params_size += dists[k][0]
            if not dists[k][1]: is_discrete = False

    return params_size, is_discrete, dists # params_size = total size, is_discrete = False if any continuous

def gym_obs_embed(obs_space, dtype):
    is_vec, sample = True, None
    # TODO expand the Dict of observation types (env.observation_space) to auto make input embedding networks
    sample = obs_space.sample()
    if isinstance(obs_space, gym.spaces.Discrete):
        sample = tf.convert_to_tensor([[sample]], dtype=dtype)
    elif isinstance(obs_space, gym.spaces.Box):
        if len(obs_space.shape) > 1: is_vec = False
        sample = tf.convert_to_tensor([sample], dtype=dtype)
    # elif isinstance(obs_space, gym.spaces.Tuple):
    elif isinstance(obs_space, gym.spaces.Dict):
        sample = gym_flatten(obs_space, sample, dtype=dtype)
        sample = tf.convert_to_tensor([sample], dtype=dtype)

    return is_vec, sample

def gym_obs_get(obs_space, x):
    if isinstance(obs_space, gym.spaces.Discrete): return np.asarray([x], obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Box): return x
    elif isinstance(obs_space, gym.spaces.Dict):
        rtn = gym_flatten(obs_space, x, dtype=obs_space.dtype)
        return rtn


def gym_action_get_mem(action_space, size):
    if isinstance(action_space, gym.spaces.Discrete): return np.zeros([size], dtype=action_space.dtype)
    elif isinstance(action_space, gym.spaces.Box): return np.zeros([size]+list(action_space.shape), dtype=action_space.dtype)
    elif isinstance(action_space, gym.spaces.Tuple):
        actions = [None]*len(action_space.spaces)
        for i,space in enumerate(action_space.spaces): actions[i] = gym_action_get_mem(space,size)
        return actions
    elif isinstance(action_space, gym.spaces.Dict):
        actions = {}
        for k,space in action_space.spaces.items(): actions[k] = gym_action_get_mem(space,size)
        return actions

def gym_obs_get_mem(obs_space, size):
    if isinstance(obs_space, gym.spaces.Discrete): return np.zeros([size]+[1], dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Box): return np.zeros([size]+list(obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        sample = obs_space.sample()
        sample = gym_flatten(obs_space, sample, dtype=obs_space.dtype)
        return np.zeros([size]+list(sample.shape), dtype=obs_space.dtype)

def update_mem(mem, index, item):
    if isinstance(mem, list):
        for i in range(len(mem)): mem[i][index] = item[i]
    elif isinstance(mem, dict):
        for k in mem.keys(): mem[k][index] = item[k]
    else: mem[index] = item



def dist_loss_entropy(dists, logits, targets, discrete=True, num_samples=1):
    # dists[0][0] = param size, dists[0][1] = is discrete, dists[0][2] = dists lambda
    if isinstance(dists, list):
        cnt, s, e = len(dists), 0, dists[0][0]
        loss, entropy = dist_loss_entropy(dists[0][2],logits[:,s:e],targets[0],dists[0][1],num_samples); s = e
        for i in range(1, cnt):
            e = s+dists[i][0]
            ls, en = dist_loss_entropy(dists[i][2],logits[:,s:e],targets[i],dists[i][1],num_samples); s = e
            loss += ls; entropy += en
        loss, entropy = loss/cnt, entropy/cnt # all loss from here is always in the same range cause it's based on the 0.0-1.0 probability
    elif isinstance(dists, dict):
        keys = list(dists.keys()); k0 = keys[0]
        cnt, s, e = len(keys), 0, dists[k0][0]
        loss, entropy = dist_loss_entropy(dists[k0][2],logits[:,s:e],targets[k0],dists[k0][1],num_samples)
        s = e
        for i in range(1, cnt):
            k = keys[i]
            e = s+dists[k][0]
            ls, en = dist_loss_entropy(dists[k][2],logits[:,s:e],targets[k],dists[k][1],num_samples)
            s = e
            loss += ls; entropy += en
        loss, entropy = loss/cnt, entropy/cnt
    else:
        dist = dists(logits)
        # dist = tfp.distributions.TransformedDistribution(distribution=dist, bijector=self.bijector)
        targets = tf.cast(targets, dtype=dist.dtype)
        # if dist.event_shape.rank == 0: targets = tf.squeeze(targets, axis=-1)
        # loss = dist.prob(targets)
        loss = -dist.log_prob(targets)
        # # PPO
        # log_prob = dist.log_prob(targets)
        # log_prob_prev = tf.roll(log_prob, shift=1, axis=0)
        # loss = tf.math.exp(log_prob - log_prob_prev)
        if discrete: entropy = dist.entropy()
        else:
            entropy = dist.sample(num_samples)
            entropy = -dist.log_prob(entropy)
            # entropy = util.replace_infnan(entropy, self.float_maxroot)
            entropy = tf.reduce_mean(entropy, axis=0)

    return loss, entropy

def dist_sample(dists, logits):
    # v[0] = param size, v[1] = is discrete, v[2] = dist lambda
    if isinstance(dists, list):
        s, action = 0, [None]*len(dists)
        for i,v in enumerate(dists):
            e = s+v[0]; action[i] = dist_sample(v[2],logits[:,s:e]); s = e
    elif isinstance(dists, dict):
        s, action = 0, {}
        for k,v in dists.items():
            e = s+v[0]; action[k] = dist_sample(v[2],logits[:,s:e]); s = e
    else:
        dist = dists(logits)
        # dist = tfp.distributions.TransformedDistribution(distribution=dist, bijector=self.bijector) # doesn't sample with float64
        action = tf.squeeze(dist.sample(), axis=0)
    return action
    
    # action = dist.sample()
    # test = {}
    # test['name'] = tf.squeeze(action, axis=0)
    # # test = [tf.squeeze(action, axis=0),]
    # return test


def tf_to_np(data):
    if isinstance(data, tf.Tensor): return data.numpy()
    elif isinstance(data, list):
        for i,v in enumerate(data): data[i] = tf_to_np(v)
    elif isinstance(data, dict):
        for k,v in data.items(): data[k] = tf_to_np(v)
    return data
def np_to_tf(data, dtype=None):
    if isinstance(data, np.ndarray): return tf.convert_to_tensor(data, dtype=dtype)
    elif isinstance(data, list):
        data_tf = [None]*len(data)
        for i,v in enumerate(data): data_tf[i] = np_to_tf(v, dtype=dtype)
    elif isinstance(data, dict):
        data_tf = {}
        for k,v in data.items(): data_tf[k] = np_to_tf(v, dtype=dtype)
    return data_tf
