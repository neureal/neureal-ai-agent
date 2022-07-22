import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numba

# TODO put this in seperate repo
# TODO test to make sure looping constructs are working like I think, ie python loop only happens once on first trace and all refs are correct on subsequent runs

def print_time(t):
    days=int(t//86400);hours=int((t-days*86400)//3600);mins=int((t-days*86400-hours*3600)//60);secs=int((t-days*86400-hours*3600-mins*60))
    return "{:4d}:{:02d}:{:02d}:{:02d}".format(days,hours,mins,secs)

def replace_infnan(inputs, replace):
    isinfnan = tf.math.logical_or(tf.math.is_nan(inputs), tf.math.is_inf(inputs))
    return tf.where(isinfnan, replace, inputs)

# TODO tf.keras.layers.Discretization ?
def discretize(inputs, spec):
    if spec['dtype'] == tf.uint8 or spec['dtype'] == tf.int32 or spec['dtype'] == tf.int64 or spec['dtype'] == tf.bool: inputs = tf.math.round(inputs)
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

def stats_update(stats_spec, value):
    b1, b1_n, b2, b2_n, dtype, var_ma, var_ema, var_iter = stats_spec['b1'], stats_spec['b1_n'], stats_spec['b2'], stats_spec['b2_n'], stats_spec['dtype'], stats_spec['ma'], stats_spec['ema'], stats_spec['iter']
    one = tf.constant(1, dtype)
    var_ma.assign(b1 * var_ma + b1_n * value)
    var_ema.assign(b2 * var_ema + b2_n * (value * value))
    var_iter.assign_add(one)
def stats_get(stats_spec):
    b1, b2, dtype, var_ma, var_ema, var_iter = stats_spec['b1'], stats_spec['b2'], stats_spec['dtype'], stats_spec['ma'], stats_spec['ema'], stats_spec['iter']
    zero, one, float_eps = tf.constant(0, dtype), tf.constant(1, dtype), tf.constant(tf.experimental.numpy.finfo(dtype).eps, dtype)

    ma = one - tf.math.pow(b1, var_iter)
    ema = one - tf.math.pow(b2, var_iter)
    if ma < float_eps: ma = float_eps
    if ema < float_eps: ema = float_eps
    ma = var_ma / ma
    ema = var_ema / ema
    ema = tf.math.sqrt(ema)
    if ema < float_eps: ema = float_eps

    snr = tf.math.abs(ma) / ema
    # std = ema - tf.math.abs(ma)
    std = ema - ma # -std2

    if ma < zero: ema = -ema
    return ma, ema, snr, std


def net_build(net, initializer):
    net.reset_states()
    net.weights_reset = []
    for w in net.weights:
        w.is_kernel = True if "/kernel:" in w.name else False
        if w.is_kernel: w.assign(initializer(w.shape))
        net.weights_reset.append(w.value())
    net.initializer = initializer

def net_reset(net):
    for i in range(len(net.weights)):
        w = net.weights[i]
        if w.is_kernel: w.assign(net.initializer(w.shape))
        else: w.assign(net.weights_reset[i])

def net_copy(source, dest):
    for i in range(len(dest.weights)): dest.weights[i].assign(source.weights[i].value())

def optimizer(net_name, opt_spec):
    beta_1, beta_2, decay = tf.constant(0.99,tf.float64), tf.constant(0.99,tf.float64), 0 # tf.constant(0.0,tf.float64)
    typ, schedule_type, learn_rate, float_eps, name = opt_spec['type'], opt_spec['schedule_type'], opt_spec['learn_rate'], opt_spec['float_eps'], '{}/optimizer_{}/'.format(net_name, opt_spec['name'])
    maxval, minval = tf.constant(learn_rate), tf.cast(float_eps,tf.float64); mean, stddev = tf.constant(maxval/2), tf.constant((maxval-minval)/4)
    def schedule_r(): return tf.random.uniform((), dtype=tf.float64, maxval=maxval, minval=minval)
    def schedule_rtn(): return tf.random.truncated_normal((), dtype=tf.float64, mean=mean, stddev=stddev)

    if schedule_type == 'r': learn_rate = schedule_r
    if schedule_type == 'rtn': learn_rate = schedule_rtn
    if schedule_type == 'cd': learn_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learn_rate, first_decay_steps=16, t_mul=1.0, m_mul=1.0, alpha=minval)
    if schedule_type == 'tc': learn_rate = tfa.optimizers.TriangularCyclicalLearningRate(initial_learning_rate=learn_rate, maximal_learning_rate=minval, step_size=16, scale_mode='cycle')
    if schedule_type == 'ex': learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learn_rate, decay_steps=opt_spec['num_steps'], decay_rate=tf.constant(opt_spec['lr_min']/learn_rate,tf.float64), staircase=False)

    if typ == 's': optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate, name=name+'SGD')
    if typ == 'a': optimizer = tf.keras.optimizers.Adam(beta_1=beta_1, beta_2=beta_2, decay=decay, amsgrad=False, learning_rate=learn_rate, epsilon=float_eps, name=name+'Adam')
    if typ == 'am': optimizer = tf.keras.optimizers.Adamax(beta_1=beta_1, beta_2=beta_2, decay=decay, learning_rate=learn_rate, epsilon=float_eps, name=name+'Adam')
    if typ == 'aw': optimizer = tfa.optimizers.AdamW(learning_rate=learn_rate, epsilon=float_eps, weight_decay=opt_spec['weight_decay'], name=name+'AdamW')
    if typ == 'ar': optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learn_rate, epsilon=float_eps, name=name+'RectifiedAdam')
    if typ == 'ab': optimizer = tfa.optimizers.AdaBelief(learning_rate=learn_rate, epsilon=float_eps, rectify=True, name=name+'AdaBelief')
    if typ == 'co': optimizer = tfa.optimizers.COCOB(alpha=100.0, use_locking=True, name=name+'COCOB')
    if typ == 'ws': optimizer = tfa.optimizers.SWA(tf.keras.optimizers.SGD(learning_rate=learn_rate), start_averaging=0, average_period=10, name=name+'SWA') # has error with floatx=float64
    if typ == 'sw': optimizer = tfa.optimizers.SGDW(learning_rate=learn_rate, weight_decay=opt_spec['weight_decay'], name=name+'SGDW')
    # if typ == 'ax': optimizer = tf.keras.optimizers.experimental.Adam(beta_1=beta_1, beta_2=beta_2, amsgrad=False, learning_rate=learn_rate, epsilon=float_eps, name=name+'AdamEx')

    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=False, initial_scale=2**15)
    if schedule_type == 'ep': optimizer.episodes = optimizer.add_slot(tf.Variable(0, trainable=False, name=name), 'episodes')
    return optimizer

def optimizer_build(optimizer, variables):
    optimizer.apply_gradients(zip(variables, variables))
    for w in optimizer.weights: w.assign(tf.zeros_like(w))
    return optimizer.weights

class LearnRateThresh():
    def __init__(self, thresh, thresh_rates):
        # thresh = [0.1,2.0]
        # thresh_rate = [71,51,31] # 2e12 107, 2e10 89, 2e8 71, 2e6 53, 2e5 44, 2e4 35, 2e3 26, 2e2 17
        self.thresh, self.thresh_rates = tf.constant(thresh,tf.float64), tf.constant(thresh_rates,tf.int32)
        float64_eps = tf.constant(tf.experimental.numpy.finfo(tf.float64).eps,tf.float64)
        x = tf.constant([9,8,7,6,5,4,3,2,1], tf.float64)
        # x = tf.constant([1], tf.float64)
        d = int(np.ceil(-np.log10(float64_eps)))
        learn_rate_cats = tf.math.pow(10.0, -tf.range(tf.constant(1,tf.float64), d+1))
        x, y = tf.meshgrid(x, learn_rate_cats)
        self.learn_rate_cats = tf.concat([tf.constant([1], tf.float64), tf.reshape(x*y,(-1))], 0)
    def __call__(self, metric):
        thresh_idx = tf.searchsorted(self.thresh, tf.reshape(tf.cast(metric,tf.float64),(1,)))
        learn_rate_idx = self.thresh_rates[thresh_idx[0]]
        learn_rate = self.learn_rate_cats[learn_rate_idx]
        return learn_rate


def loss_diff(out, targets=None): # deterministic difference
    compute_dtype = tf.keras.backend.floatx()
    if isinstance(out, list):
        loss = tf.constant(0, compute_dtype)
        for i in range(len(out)):
            o, t = tf.cast(out[i], compute_dtype), tf.cast(targets[i], compute_dtype)
            loss = loss + tf.math.abs(tf.math.subtract(o, t)) # MAE
            # loss = loss + tf.math.square(tf.math.subtract(o, t)) # MSE
    else:
        out = tf.cast(out, compute_dtype)
        if targets is None: diff = out
        else:
            targets = tf.cast(targets, compute_dtype)
            diff = tf.math.subtract(out, targets)
        # loss = tf.where(tf.math.less(diff, compute_zero), tf.math.negative(diff), diff) # MAE
        loss = tf.math.abs(diff) # MAE
        # loss = tf.math.square(diff) # MSE
    loss = tf.math.reduce_sum(loss, axis=tf.range(1, tf.rank(loss)))
    return loss

def loss_likelihood(dist, targets, probs=False):
    if isinstance(dist, list):
        loss = tf.constant(0, tf.keras.backend.floatx())
        for i in range(len(dist)):
            t = tf.cast(targets[i], dist[i].dtype)
            if probs: loss = loss - tf.math.exp(dist[i].log_prob(t))
            else: loss = loss - tf.reduce_sum(dist[i].log_prob(t))
    else:
        targets = tf.cast(targets, dist.dtype)
        if probs: loss = -tf.math.exp(dist.log_prob(targets))
        else: loss = -tf.reduce_sum(dist.log_prob(targets))

    isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
    if isinfnan > 0: tf.print('NaN/Inf likelihood loss:', loss)
    return tf.reshape(loss,(1,))

def loss_bound(dist, targets):
    loss = -loss_likelihood(dist, targets)
    # if not categorical: loss = loss - dist_prior.log_prob(targets)

    isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
    if isinfnan > 0: tf.print('NaN/Inf bound loss:', loss)
    return loss

def loss_entropy(dist, entropy_contrib): # "Soft Actor Critic" = try increase entropy
    loss = tf.constant(0, tf.keras.backend.floatx())
    if entropy_contrib > 0.0:
        if isinstance(dist, list):
            for i in range(len(dist)): loss = loss + dist[i].entropy()
        else: loss = dist.entropy()
        loss = -loss * entropy_contrib

    isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
    if isinfnan > 0: tf.print('NaN/Inf entropy loss:', loss)
    return loss

def loss_PG(dist, targets, returns, values=None, returns_target=None): # policy gradient, actor/critic
    compute_dtype = tf.keras.backend.floatx()
    returns = tf.squeeze(tf.cast(returns, compute_dtype), axis=-1)
    loss_lik = loss_likelihood(dist, targets, probs=False)
    # loss_lik = loss_lik -float_maxroot # -float_maxroot, +float_log_min_prob, -np.e*17.0, -154.0, -308.0
    if returns_target is not None:
        returns_target = tf.squeeze(tf.cast(returns_target, compute_dtype), axis=-1)
        returns = returns_target - returns # _lRt
        # returns = returns - returns_target # _lRtn
        # returns = tf.abs(returns_target - returns) / returns_target # _lRtan
        # returns = tf.abs(returns_target - returns) # _lRta
    if values is not None: returns = returns - tf.squeeze(tf.cast(values, compute_dtype), axis=-1)
    loss = loss_lik * returns # / float_maxroot

    isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
    if isinfnan > 0: tf.print('NaN/Inf PG loss:', loss)
    return loss



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

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = tf.concat([input_shape[:self._axis], self._group_shape], axis=0)
        grouped_inputs = tf.reshape(inputs, group_shape)
        _, var = tf.nn.moments(grouped_inputs, self._std_shape, keepdims=True)
        std = tf.sqrt(var + self._eps)
        std = tf.broadcast_to(std, group_shape)
        group_std = tf.reshape(std, input_shape)

        return (inputs * tf.math.sigmoid(self._v1 * inputs)) / group_std * self._gamma + self._beta



def distribution(dist_spec):
    dist_type, num_components, event_shape = dist_spec['dist_type'], dist_spec['num_components'], dist_spec['event_shape']
    if dist_type == 'd': params_size, dist = Deterministic.params_size(event_shape), Deterministic(event_shape)
    if dist_type == 'c': params_size, dist = Categorical.params_size(num_components, event_shape), Categorical(num_components, event_shape)
    # if dist_type == 'c': params_size, dist = CategoricalRP.params_size(event_shape), CategoricalRP(event_shape)
    if dist_type == 'mx': params_size, dist = MixtureLogistic.params_size(num_components, event_shape), MixtureLogistic(num_components, event_shape)
    # if dist_type == 'mx': params_size, dist = tfp.layers.MixtureLogistic.params_size(num_components, event_shape), tfp.layers.MixtureLogistic(num_components, event_shape) # makes NaNs
    # if dist_type == 'mt': params_size, dist = MixtureMultiNormalTriL.params_size(num_components, event_shape, matrix_size=2), MixtureMultiNormalTriL(num_components, event_shape, matrix_size=2)
    return params_size, dist

class DeterministicSub(tfp.distributions.Deterministic):
    def _log_prob(self, x):
        # return tf.constant([-1], dtype=x.dtype)
        loc = tf.convert_to_tensor(self.loc)
        loss = tf.math.abs(tf.math.subtract(x, loc))
        loss = tf.math.negative(tf.math.reduce_sum(loss, axis=tf.range(1, tf.rank(loss))))
        return loss
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
        # dist = tfp.distributions.Deterministic(loc=params)
        dist = DeterministicSub(loc=params)
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
        # params = tf.clip_by_value(params, -1, 1) # _cat-clip
        # params = tfp.math.clip_by_value_preserve_gradient(params, -1, 1) # _cat-clip-tfp
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

        # scale_params = tf.math.abs(scale_params)
        # # scale_params = tfp.math.clip_by_value_preserve_gradient(scale_params, eps, maxroot)
        # scale_params = tf.clip_by_value(scale_params, eps, maxroot)
        sharpness = tf.constant(1e-2, scale_params.dtype)
        scale_params = tf.math.softplus(scale_params*sharpness)/sharpness
        # scale_params = tf.math.softplus(scale_params)
        scale_params = scale_params + eps
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
        key_dim = int(channels/num_heads) if cross_type == 2 else int(latent_size/num_heads)
        # key_dim = int(latent_size/num_heads)
        super(MultiHeadAttention, self).__init__(tf.identity(num_heads), tf.identity(key_dim), use_bias=use_bias, **kwargs)
        self._mem_size, self._sort_memory, self._norm, self._residual, self._cross_type = memory_size, sort_memory, norm, residual, cross_type
        self._mem_channels = latent_size if cross_type != 1 else channels
        float_eps = tf.experimental.numpy.finfo(self.compute_dtype).eps

        if norm:
            # self._layer_norm_key = tf.keras.layers.LayerNormalization(epsilon=float_eps, center=True, scale=True, name='norm_key')
            # self._layer_norm_value = tf.keras.layers.LayerNormalization(epsilon=float_eps, center=True, scale=True, name='norm_value')
            self._layer_dense_key_in = tf.keras.layers.Dense(hidden_size, activation=EvoNormS0(evo), use_bias=False, name='dense_key_in')
            self._layer_dense_value_in = tf.keras.layers.Dense(hidden_size, activation=EvoNormS0(evo), use_bias=False, name='dense_value_in')

        # query_scale = 1.0 / tf.math.sqrt(tf.cast(self._key_dim, dtype=self.compute_dtype))
        # self._query_scale = tf.identity(query_scale)

        if cross_type: # CrossAttention
            if init_zero is None:
                # init_zero = tf.random.normal((1, num_latents, latent_size), mean=0.0, stddev=0.02, dtype=self.compute_dtype)
                # init_zero = tf.clip_by_value(init_zero, -2.0, 2.0)
                # init_zero = np.random.uniform(np.asarray(-1.0, dtype=self.compute_dtype), 1.0, num_latents)
                if cross_type == 1: # input, batch = different latents
                    init_zero = np.linspace(1.0, -1.0, num_latents, dtype=self.compute_dtype) # -np.e, np.e
                    init_zero = np.expand_dims(init_zero, axis=-1)
                    init_zero = np.repeat(init_zero, latent_size, axis=-1)
                if cross_type == 2: # output, batch = actual batch
                    init_zero = np.linspace(1.0, -1.0, channels, dtype=self.compute_dtype) # -np.e, np.e
                    init_zero = np.expand_dims(init_zero, axis=0)
                    init_zero = np.repeat(init_zero, num_latents, axis=0)
                init_zero += float_eps
                init_zero = np.expand_dims(init_zero, axis=0)
            init_zero = tf.constant(init_zero, dtype=self.compute_dtype)
            self._init_zero = tf.identity(init_zero)

    def build(self, input_shape):
        if self._mem_size is not None:
            mem_shape = (self._mem_size, input_shape[-2], input_shape[-1]); mem_zero = tf.constant(np.full(mem_shape, 0), self.compute_dtype)
            # mem_rel_idx = np.linspace(np.full(mem_shape[1:], self._mem_size), np.full(mem_shape[1:], 0), self._mem_size+2, dtype=self.compute_dtype)[1:-1]
            # img_rel_idx = np.linspace(np.full(mem_shape[1:], 0), np.full(mem_shape[1:], -1), self._mem_size+2, dtype=self.compute_dtype)[1:-1]
            mem_rel_idx = np.linspace(np.full((input_shape[-2],1), 1), np.full((input_shape[-2],1), 0), self._mem_size+2, dtype=self.compute_dtype)[1:-1]
            img_rel_idx = np.linspace(np.full((input_shape[-2],1), 0), np.full((input_shape[-2],1), -1), self._mem_size+2, dtype=self.compute_dtype)[1:-1]
            mem_rel_idx, value_rel_idx, img_rel_idx = tf.constant(mem_rel_idx, self.compute_dtype), tf.zeros((1,input_shape[-2],1), self.compute_dtype), tf.constant(img_rel_idx, self.compute_dtype)
            self._mem_zero, self._mem_rel_idx, self._value_rel_idx, self._img_rel_idx = tf.identity(mem_zero), tf.identity(mem_rel_idx), tf.identity(value_rel_idx), tf.identity(img_rel_idx)
            self._mem_idx, self._memory = tf.Variable(self._mem_size, trainable=False, name='mem_idx'), tf.Variable(self._mem_zero, trainable=False, name='memory')
            self._mem_idx_img, self._memory_img = tf.Variable(self._mem_size, trainable=False, name='mem_idx_img'), tf.Variable(self._mem_zero, trainable=False, name='memory_img')
            # if self._sort_memory:
            #     mem_score_zero = tf.constant(np.full((self._mem_size, input_shape[-2]), 0), self.compute_dtype)
            #     self._mem_score_zero = tf.identity(mem_score_zero)
            #     self._mem_score = tf.Variable(self._mem_score_zero, trainable=False, name='mem_score')
        if self._cross_type: self._init_latent = tf.Variable(self._init_zero, trainable=True, name='init_latent')
        if self._residual: self._residual_amt = tf.Variable(0.0, dtype=self.compute_dtype, trainable=True, name='residual') # ReZero

    def _compute_attention(self, query, key, value, attention_mask=None):
        # query = tf.math.multiply(query, self._query_scale)
        attn_scores = special_math_ops.einsum(self._dot_product_equation, key, query)
        attn_scores = self._masked_softmax(attn_scores, attention_mask) # TODO can I replace softmax here with somthing more log likelihood related? (ie continuous attn)
        attn_output = special_math_ops.einsum(self._combine_equation, attn_scores, value)
        return attn_output, attn_scores

    def call(self, value, attention_mask=None, auto_mask=None, store_memory=True, use_img=False, store_real=False, num_latents=None):
        latent_size = tf.shape(value)[-1]
        if self._cross_type:
            # value[0](batch) = time dim, value[1:-2] = space/feat dim, value[-1] = channel dim
            value = tf.reshape(value, (1, -1, latent_size))
            query = self._init_latent if num_latents is None else self._init_latent[:,:num_latents]
        else: query = value
        batch_size = 1; neg_batch_size = -1; time_size = tf.shape(value)[1]
        # batch_size = tf.shape(value)[0]; neg_batch_size = -batch_size; time_size = tf.shape(value)[1]
        if not self._built_from_signature: self._build_from_signature(query=query, value=value, key=None)

        if self._mem_size is not None:
            # value_mem = memory[:,mem_idx:] # gradients not always working with this
            memory, mem_idx, memory_img, mem_idx_img, mem_size = self._memory, self._mem_idx, self._memory_img, self._mem_idx_img, self._mem_size
            # TODO add mem/img index relative to current location (mem-,img+) # tf.linspace(0.0,1,256)
            if use_img and store_real:
                # value_mem = tf.concat([memory[mem_idx:], value, memory_img[mem_idx_img+batch_size:]], axis=0)
                # value_mem = tf.reshape(value_mem, (1, -1, latent_size))

                # # value_mem = tf.concat([memory[mem_idx:]*self._mem_rel_idx[mem_idx:], value, memory_img[mem_idx_img+batch_size:]*self._img_rel_idx[mem_size-mem_idx_img:]], axis=0) # TODO

                memory_ri = tf.concat([memory[mem_idx:], self._mem_rel_idx[mem_idx:]], axis=-1)
                value_ri = tf.concat([value, self._value_rel_idx], axis=-1)
                memory_img_ri = tf.concat([memory_img[mem_idx_img+batch_size:], self._img_rel_idx[batch_size:mem_size-mem_idx_img]], axis=-1)
                value_mem = tf.concat([memory_ri, value_ri, memory_img_ri], axis=0)
                value_mem = tf.reshape(value_mem, (1, -1, latent_size+1))
            elif use_img:
                # mem = tf.concat([memory[mem_idx+(mem_size-mem_idx_img):], memory_img[mem_idx_img:]], axis=0) # TODO
                clip = mem_size-mem_idx-mem_idx_img
                if clip < 0: clip = tf.constant(0)
                value_mem =  tf.concat([memory[mem_idx+clip:], memory_img[mem_idx_img:], value], axis=0)
                value_mem = tf.reshape(value_mem, (1, -1, latent_size))
            else:
                # value_mem = tf.concat([memory[mem_idx:], value], axis=0)
                # value_mem = tf.reshape(value_mem, (1, -1, latent_size))

                # if mem_idx < mem_size:
                #     # mem = memory[mem_idx:] * self._mem_rel_idx[mem_idx:] # _mha_relM
                #     mem = memory[mem_idx:] + self._mem_rel_idx[mem_idx:] # _mha_relP
                #     value_mem = tf.concat([mem, value], axis=0)
                # else: value_mem = value

                memory_ri = tf.concat([memory[mem_idx:], self._mem_rel_idx[mem_idx:]], axis=-1)
                value_ri = tf.concat([value, self._value_rel_idx], axis=-1)
                value_mem = tf.concat([memory_ri, value_ri], axis=0)
                value_mem = tf.reshape(value_mem, (1, -1, latent_size+1))
            # query = tf.reshape(query, (1, -1, latent_size))
        else: value_mem = value

        if self._mem_size is not None and store_memory:
            if use_img and not store_real: mem_idx, memory = self._mem_idx_img, self._memory_img
            else: mem_idx, memory = self._mem_idx, self._memory

            drop_off = tf.roll(memory, shift=neg_batch_size, axis=0)
            memory.assign(drop_off)
            memory[neg_batch_size:].assign(value)

            if mem_idx > 0:
                mem_idx_next = mem_idx - batch_size
                if mem_idx_next < 0: mem_idx_next = 0
                mem_idx.assign(mem_idx_next)

        # TODO loop through value or query if too big for memory?
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

        # if self._mem_size is not None and store_memory and self._sort_memory and (store_real or not use_img):
        #     drop_off = tf.roll(self._mem_score, shift=-time_size, axis=1)
        #     self._mem_score.assign(drop_off)
        #     zero = self._mem_score_zero[:,-time_size:]
        #     self._mem_score[:,-time_size:].assign(zero)

        #     # scores = tf.math.reduce_sum(attn_scores, axis=(1,2))[0][:self._mem_size-self._mem_idx]
        #     # scores = tf.argsort(scores, axis=-1, direction='ASCENDING', stable=True)
        #     # scores = tf.gather(self._memory[:,self._mem_idx:], scores, axis=1)
        #     # self._memory[:,self._mem_idx:].assign(scores)

        #     scores = tf.math.reduce_mean(attn_scores, axis=(1,2))[0][:self._mem_size-self._mem_idx] # heads
        #     # norm = tf.cast(tf.shape(scores)[0], dtype=self.compute_dtype)
        #     # scores = tf.math.multiply(scores, norm)
        #     scores = tf.math.add(self._mem_score[:,self._mem_idx:], scores)
        #     self._mem_score[:,self._mem_idx:].assign(scores)

        #     scores = tf.argsort(scores[0], axis=-1, direction='ASCENDING', stable=True)

        #     mem_sorted = tf.gather(self._memory[:,self._mem_idx:], scores, axis=1)
        #     self._memory[:,self._mem_idx:].assign(mem_sorted)

        #     scores_sorted = tf.gather(self._mem_score[:,self._mem_idx:], scores, axis=1)
        #     self._mem_score[:,self._mem_idx:].assign(scores_sorted)

        attn_output = self._output_dense(attn_output)
        if self._residual: attn_output = query + attn_output * self._residual_amt # ReZero
        if self._cross_type: attn_output = tf.squeeze(attn_output, axis=0)
        return attn_output

    def reset_states(self, use_img=False):
        if self._mem_size is not None:
            if use_img:
                self._mem_idx_img.assign(self._mem_size)
                self._memory_img.assign(self._mem_zero)
            else:
                self._mem_idx.assign(self._mem_size)
                self._memory.assign(self._mem_zero)
                # if self._sort_memory: self._mem_score.assign(self._mem_score_zero)



class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, latent_size, evo=None, residual=True, **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        if evo is None: self._layer_dense = tf.keras.layers.Dense(hidden_size, activation=tf.keras.activations.gelu, use_bias=False, name='dense')
        else: self._layer_dense = tf.keras.layers.Dense(hidden_size, activation=EvoNormS0(evo), use_bias=False, name='dense')
        self._layer_dense_latent = tf.keras.layers.Dense(latent_size, name='dense_latent')
        self._residual = residual

    def build(self, input_shape):
        if self._residual: self._residual_amt = tf.Variable(0.0, dtype=self.compute_dtype, trainable=True, name='residual') # ReZero

    def call(self, input):
        out = self._layer_dense(input)
        out = self._layer_dense_latent(out)
        if self._residual: out = input + out * self._residual_amt # Rezero
        return out
