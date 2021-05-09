import time, os, talib
import multiprocessing as mp
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
# tf.random.set_seed(0)
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import model_util as util
import gym

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

# TODO test conditioning with action
# TODO add RepNet
# TODO try out MuZero-ish architecture
# TODO add imagination by looping through TransNet seperately from looping through env.step

# TODO add in generic gym stuff from model_util
# TODO try embedding all the net imputs to latent_size and use # out = tf.math.accumulate_n(out_accu)
# TODO combine all the diff architectures into one and use # if arch == 'NAME': or # if arch in ('NAME1', 'NAME2'):
# TODO wrap env in seperate process and make run async with random NOOP skips to test latency learning/timing
# TODO put actor in seperate process so can run async
# TODO add GenNet and DisNet for World Model (WM)
# TODO use attention (transformer decoder) layers instead of LSTM

# TODO how to incorporate ARS random policy search?
# TODO try out the 'lottery ticket hypothosis' pruning during training
# TODO use numba to make things faster on CPU


class RepNet(tf.keras.Model):
    def __init__(self, latent_size, categorical):
        super(RepNet, self).__init__()
        self.categorical, event_shape = categorical, (latent_size,)

        if categorical: num_components = 256; params_size, self.dist = util.CategoricalRP.params_size(num_components, event_shape), util.CategoricalRP(num_components, event_shape)
        else: num_components = latent_size; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)
        # TODO add deterministic w/ latent_size output

        self.net_blocks, self.net_lstm, inp, mid, evo = 1, False, latent_size*4, latent_size*4, latent_size
        self.net_arch = "RN[inD{}-{:02d}{}D{}-cmp{}-lat{}]".format(inp, self.net_blocks, ('LS+' if self.net_lstm else ''), mid, num_components, latent_size)
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_latent = tf.keras.layers.Dense(latent_size, name='dense_latent')

        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))

        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

    def _net(self, inputs):
        out = self.layer_dense_in(inputs)
        for i in range(self.net_blocks):
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_logits_out(out)
        return out

    def reset_states(self):
        if self.net_lstm:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out = self.layer_dense_in(out)
                out = self.layer_dense_latent(out)
                out_accu.append(out)
            # if k == 'rewards':
            #     out = tf.cast(v, self.compute_dtype)
            #     out_accu.append(out)
            # if k == 'dones':
            #     out = tf.cast(v, self.compute_dtype)
            #     out_accu.append(out)
        out = tf.math.accumulate_n(out_accu)
        
        for i in range(self.net_blocks):
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_logits_out(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('rep net out:', out)
        return out


# transition dynamics within latent space
class TransNet(tf.keras.Model):
    def __init__(self, latent_size, categorical, memory_size=None):
        super(TransNet, self).__init__()
        self.categorical, event_shape = categorical, (latent_size,)

        if categorical: num_components = 256; params_size, self.dist = util.CategoricalRP.params_size(num_components, event_shape), util.CategoricalRP(num_components, event_shape)
        else: num_components = latent_size; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)

        self.net_blocks, self.net_attn, self.net_lstm, inp, mid, evo, num_heads = 2, True, False, latent_size*4, latent_size*4, latent_size, 2
        self.net_arch = "TN[inD{}-{:02d}{}{}D{}-cmp{}-lat{}-hds{}]".format(inp, self.net_blocks, ('AT+' if self.net_attn else ''), ('LS' if self.net_lstm else ''), mid, num_components, latent_size, num_heads)
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_cond_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='cond_dense_in') # action conditioning embedding
        self.layer_cond_dense_latent = tf.keras.layers.Dense(latent_size, name='cond_dense_latent')

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_latent = tf.keras.layers.Dense(latent_size, name='dense_latent')

        self.layer_attn, self.layer_lstm, self.layer_dense, self.layer_dense_lat = [], [], [], []
        for i in range(self.net_blocks):
            if self.net_attn: self.layer_attn.append(util.MultiHeadAttention(num_heads=num_heads, latent_size=latent_size, memory_size=memory_size, name='attn_{:02d}'.format(i)))
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            else: self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
            self.layer_dense_lat.append(tf.keras.layers.Dense(latent_size, name='dense_lat_{:02d}'.format(i)))
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

    def reset_states(self):
        if self.net_attn:
            for i in range(self.net_blocks): self.layer_attn[i].reset_states()
        if self.net_lstm:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'actions':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out = self.layer_cond_dense_in(out)
                out = self.layer_cond_dense_latent(out)
                out_accu.append(out)
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out = self.layer_dense_in(out)
                out = self.layer_dense_latent(out)
                out_accu.append(out)
        out = tf.math.accumulate_n(out_accu)
        
        for i in range(self.net_blocks):
            out = tf.expand_dims(out, axis=0)
            if self.net_attn: out = self.layer_attn[i](out, training=training)
            if self.net_lstm: out = self.layer_lstm[i](out, training=training)
            out = tf.squeeze(out, axis=0)
            if not self.net_lstm: out = self.layer_dense[i](out)
            out = self.layer_dense_lat[i](out)
        out = self.layer_dense_logits_out(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('trans net out:', out)
        return out


class RewardNet(tf.keras.Model):
    def __init__(self):
        super(RewardNet, self).__init__()
        num_components, event_shape = 16, (1,); params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)
        inp, evo = 256, 32; self.net_arch = "RWD[inD{}-cmp{}]".format(inp, num_components)
        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        # self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
    @tf.function
    def call(self, inputs, training=None):
        out = self.layer_flatten(inputs['obs'])
        out = self.layer_dense_in(out)
        out = self.layer_dense_logits_out(out)
        return out

class DoneNet(tf.keras.Model):
    def __init__(self):
        super(DoneNet, self).__init__()
        num_components, event_shape = 2, (1,); params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        # params_size, self.dist = 1, tfp.layers.DistributionLambda(lambda input: tfp.distributions.Bernoulli(logits=input, dtype=tf.bool))
        inp, evo = 256, 32; self.net_arch = "DON[inD{}-cmp{}]".format(inp, num_components)
        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        # self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
    @tf.function
    def call(self, inputs, training=None):
        out = self.layer_flatten(inputs['obs'])
        out = self.layer_dense_in(out)
        out = self.layer_dense_logits_out(out)
        return out

class GenNet(tf.keras.Model):
    def __init__(self, spec, force_cont, latent_size):
        super(GenNet, self).__init__()

        num_components, event_shape = spec['num_components'], spec['event_shape']
        if not force_cont and spec['is_discrete']: params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        else: num_components *= 4; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)

        self.net_blocks, self.net_lstm, inp, mid, evo = 1, False, latent_size*4, latent_size*4, int(latent_size/2)
        self.net_arch = "GN[inD{}-{:02d}{}D{}-cmp{}{}]".format(inp, self.net_blocks, ('LS+' if self.net_lstm else ''), mid, num_components, ('-con' if force_cont else ''))
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
        
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

    def reset_states(self):
        if self.net_lstm:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
        out = tf.concat(out_accu, 1)
        out = self.layer_dense_in(out)
        for i in range(self.net_blocks):
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_logits_out(out)
        
        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('gen net out:', out)
        return out


class ActionNet(tf.keras.Model):
    def __init__(self, spec, force_cont, latent_size, memory_size=None):
        super(ActionNet, self).__init__()

        num_components, event_shape = spec['num_components'], spec['event_shape']
        if not force_cont and spec['is_discrete']: params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        else: num_components *= 4; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)

        self.net_blocks, self.net_attn, self.net_lstm, inp, mid, evo, num_heads = 1, False, False, latent_size*4, latent_size*4, int(latent_size/2), 2
        self.net_arch = "AN[inD{}-{:02d}{}{}D{}-cmp{}{}{}]".format(inp, self.net_blocks, ('AT+' if self.net_attn else ''), ('LS+' if self.net_lstm else ''), mid, num_components, ('-con' if force_cont else ''), ('-hds'+str(num_heads) if self.net_attn else ''))
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_latent = tf.keras.layers.Dense(latent_size, name='dense_latent')

        self.layer_attn, self.layer_lstm, self.layer_dense, self.layer_dense_lat = [], [], [], []
        for i in range(self.net_blocks):
            if self.net_attn: self.layer_attn.append(util.MultiHeadAttention(num_heads=num_heads, latent_size=latent_size, memory_size=memory_size, name='attn_{:02d}'.format(i)))
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            else: self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
            self.layer_dense_lat.append(tf.keras.layers.Dense(latent_size, name='dense_lat_{:02d}'.format(i)))
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

    def reset_states(self):
        if self.net_attn:
            for i in range(self.net_blocks): self.layer_attn[i].reset_states()
        if self.net_lstm:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out = self.layer_dense_in(out)
                out = self.layer_dense_latent(out)
                out_accu.append(out)
        out = tf.math.accumulate_n(out_accu)
        
        for i in range(self.net_blocks):
            out = tf.expand_dims(out, axis=0)
            if self.net_attn: out = self.layer_attn[i](out, training=training)
            if self.net_lstm: out = self.layer_lstm[i](out, training=training)
            out = tf.squeeze(out, axis=0)
            if not self.net_lstm: out = self.layer_dense[i](out)
            out = self.layer_dense_lat[i](out)
        out = self.layer_dense_logits_out(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('action net out:', out)
        return out


class ValueNet(tf.keras.Model):
    def __init__(self, latent_size):
        super(ValueNet, self).__init__()
        self.net_blocks, self.net_lstm, inp, mid, evo = 1, False, latent_size*2, latent_size*2, int(latent_size/2)
        self.net_arch = "VN[inD{}-{:02d}{}D{}]".format(inp, self.net_blocks, ('LS+' if self.net_lstm else ''), mid)
        self.layer_flatten = tf.keras.layers.Flatten()

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_lstm: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
        self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')

    def reset_states(self):
        if self.net_lstm:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
        out = tf.concat(out_accu, 1)
        out = self.layer_dense_in(out)
        for i in range(self.net_blocks):
            if self.net_lstm: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_out(out)
        return out


class GeneralAI(tf.keras.Model):
    def __init__(self, arch, env, render, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, force_cont_ob, force_cont_action, latent_size, latent_cat, memory_size):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(compute_dtype.max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)

        self.arch, self.env, self.render, self.force_cont_ob, self.force_cont_action = arch, env, render, force_cont_ob, force_cont_action
        self.max_episodes, self.max_steps, self.entropy_contrib, self.returns_disc = tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(entropy_contrib, compute_dtype), tf.constant(returns_disc, tf.float64)
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)

        self.ob_spec, self.ob_zero, self.ob_zero_out = util.gym_get_spec(env.observation_space, self.compute_dtype, force_cont=force_cont_ob, rtn_tf=True)
        self.action_spec, self.action_zero, self.action_zero_out = util.gym_get_spec(env.action_space, self.compute_dtype, force_cont=force_cont_action, rtn_tf=True)

        self.latent_dtype = compute_dtype
        # self.latent_dtype = tf.int64 if latent_cat else compute_dtype
        # self.latent_zero, self.latent_invar = tf.constant(0, self.latent_dtype, shape=(1,latent_size)), tf.TensorShape([None,latent_size])
        # self.action_invar = tf.TensorShape([None]+action_shape)


        inputs = {'obs':self.ob_zero, 'rewards':tf.constant([[0]],tf.float64), 'dones':tf.constant([[False]],tf.bool)}
        # self.obs_zero = tf.concat([inputs['obs'], tf.cast(inputs['rewards'],self.obs_dtype), tf.cast(inputs['dones'],self.obs_dtype)], axis=1)
        # inputs = {'obs':self.ob_zero}
        if arch in ('TEST','TRANS'):
            self.rep = RepNet(latent_size, latent_cat); outputs = self.rep(inputs); rep_dist = self.rep.dist(outputs)
            smpl = rep_dist.sample()
            smpl = tf.zeros_like(smpl, self.latent_dtype)
            self.latent_zero = smpl
            inputs['obs'] = self.latent_zero

        if arch in ('TEST'):
            self.gen = GenNet(self.ob_spec, force_cont_ob, latent_size); outputs = self.gen(inputs)
            self.rwd = RewardNet(); outputs = self.rwd(inputs)
            self.done = DoneNet(); outputs = self.done(inputs)
        self.action = ActionNet(self.action_spec, force_cont_action, latent_size, memory_size); outputs = self.action(inputs)
        if arch in ('TEST','AC'): self.value = ValueNet(latent_size); outputs = self.value(inputs)

        if arch in ('TEST','TRANS'):
            inputs['actions'] = self.action_zero_out
            self.trans = TransNet(latent_size, latent_cat, memory_size); outputs = self.trans(inputs)

        self.reset_states()
        self.discounted_sum = tf.Variable(0, dtype=tf.float64, trainable=False)

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=self.float_eps)


        metrics = {'rewards_total':np.float64,'rewards_final':np.float64,'steps':np.int64}
        metrics_loss = [{'loss_total':np.float64}]
        if arch in ('AC'):
            metrics_loss.append({'loss_action':np.float64,'loss_value':np.float64})
            metrics_loss.append({'returns':np.float64,'advantages':np.float64})
        # if arch in ('TEST'):
            # metrics_loss.append({'loss_rep':np.float64})
            # metrics_loss.append({'loss_rep':np.float64,'loss_trans':np.float64})
            # metrics_loss.append({'loss_rwd':np.float64,'loss_done':np.float64})

        for k in metrics.keys(): metrics[k] = np.zeros((max_episodes), metrics[k])
        for loss_group in metrics_loss:
            for k in loss_group.keys(): loss_group[k] = [[] for i in range(max_episodes)]
        self.metrics_main, self.metrics_loss = metrics, metrics_loss


    def metrics_update(self, episode, rewards_total, rewards_final, steps, loss_total, loss_action, loss_value, returns, advantages, loss_rep, loss_trans, loss_rwd, loss_done):
        episode, rewards_total, rewards_final, steps, loss_total, loss_action, loss_value, returns, advantages, loss_rep, loss_trans, loss_rwd, loss_done = \
            episode.item(), rewards_total.item(), rewards_final.item(), steps.item(), loss_total.item(), loss_action.item(), loss_value.item(), returns.item(), advantages.item(), loss_rep.item(), loss_trans.item(), loss_rwd.item(), loss_done.item()
        self.metrics_main['rewards_total'][episode] += rewards_total
        self.metrics_main['rewards_final'][episode] = rewards_final
        self.metrics_main['steps'][episode] += steps
        if loss_total is not False: self.metrics_loss[0]['loss_total'][episode].append(loss_total)
        if loss_action is not False: self.metrics_loss[1]['loss_action'][episode].append(loss_action)
        if loss_value is not False: self.metrics_loss[1]['loss_value'][episode].append(loss_value)
        if returns is not False: self.metrics_loss[2]['returns'][episode].append(returns)
        if advantages is not False: self.metrics_loss[2]['advantages'][episode].append(advantages)
        if loss_rep is not False: self.metrics_loss[3]['loss_rep'][episode].append(loss_rep)
        if loss_trans is not False: self.metrics_loss[3]['loss_trans'][episode].append(loss_trans)
        if loss_rwd is not False: self.metrics_loss[4]['loss_rwd'][episode].append(loss_rwd)
        if loss_done is not False: self.metrics_loss[4]['loss_done'][episode].append(loss_done)


    def loss_likelihood(self, dist, targets):
        targets = tf.cast(targets, dist.dtype)
        loss = -dist.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf loss:', loss)
        return loss
        
    def loss_bound(self, dist, targets):
        targets = tf.cast(targets, dist.dtype)
        loss = dist.log_prob(targets)
        # if not self.categorical:
        # loss = loss - self.dist_prior.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf loss:', loss)
        return loss
        
    def loss_AC(self, dist, targets, advantages): # actor/critic
        targets = tf.cast(targets, dist.dtype)
        loss = -dist.log_prob(targets)
        loss = loss * advantages # * 1e-2
        # if self.categorical:
        #     entropy = dist.entropy()
        #     loss = loss - entropy * self.entropy_contrib # "Soft Actor Critic" = try increase entropy

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('NaN/Inf loss:', loss)
        return loss
        
    def loss_diff(self, diff): # deterministic difference
        # loss = tf.where(tf.math.less(advantages, 0.0), tf.math.negative(advantages), advantages) # MAE
        loss = tf.math.abs(diff) # MAE
        return loss


    def env_reset(self):
        obs, reward, done = self.env.reset(), 0.0, False
        if self.render: env.render()
        return np.expand_dims(obs,0), np.expand_dims(np.asarray(reward, np.float64),(0,1)), np.expand_dims(np.asarray(done, np.bool),(0,1))
    def env_step(self, action):
        obs, reward, done, _ = self.env.step(action)
        if self.render: env.render()
        return np.expand_dims(obs,0), np.expand_dims(np.asarray(reward, np.float64),(0,1)), np.expand_dims(np.asarray(done, np.bool),(0,1))

    @tf.function
    def reset_states(self):
        for net in self.layers:
            if hasattr(net, 'reset_states'): net.reset_states()



    @tf.function
    def TRANS_actor(self, inputs_):
        print("tracing -> GeneralAI TRANS_actor")
        inputs, outputs = inputs_.copy(), {}

        latents_next = tf.TensorArray(self.latent_dtype, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        targets = tf.TensorArray(self.ob_spec['dtype'], size=0, dynamic_size=True)

        inputs_rep = {'obs':self.latent_zero, 'actions':self.action_zero_out}
        for step in tf.range(self.max_steps):

            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs_rep['obs'] = rep_dist.sample()

            trans_logits = self.trans(inputs_rep); trans_dist = self.trans.dist(trans_logits)
            inputs_rep['obs'] = trans_dist.sample()
            latents_next = latents_next.write(step, inputs_rep['obs'][-1])

            action_logits = self.action(inputs_rep); action_dist = self.action.dist(action_logits)
            action = action_dist.sample()
            inputs_rep['actions'] = action

            if self.force_cont_action and self.action_spec['is_discrete']: action = util.discretize(action, self.action_spec['min'], self.action_spec['max'])
            action = tf.cast(action, self.action_spec['dtype'])
            action = tf.squeeze(action)
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_step, [action], (self.ob_spec['dtype'], tf.float64, tf.bool))

            rewards = rewards.write(step, inputs['rewards'][-1])
            targets = targets.write(step, inputs['obs'][-1])
            if inputs['dones'][-1][0]: break

        outputs['obs'], outputs['rewards'], outputs['targets'] = latents_next.stack(), rewards.stack(), targets.stack()
        return outputs, inputs

    @tf.function
    def TRANS_learner(self, inputs, training=True):
        print("tracing -> GeneralAI TRANS_learner")

        action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)

        loss = {}
        loss['action'] = self.loss_likelihood(action_dist, inputs['targets'])
        loss['total'] = loss['action']
        return loss

    @tf.function
    def TRANS_run(self):
        print("tracing -> GeneralAI TRANS_run")
        for episode in tf.range(self.max_episodes):
            inputs = {}
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_reset, [], (self.ob_spec['dtype'], tf.float64, tf.bool))
            while not inputs['dones'][-1][0]:
                with tf.GradientTape() as tape:
                    outputs, inputs = self.TRANS_actor(inputs)
                    loss = self.TRANS_learner(outputs)
                gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.action.trainable_variables))

                metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                    tf.math.reduce_mean(loss['total']), False, False, False, False,
                    False, False, False, False
                ]
                tf.numpy_function(self.metrics_update, metrics, ())


    @tf.function
    def AC_actor(self, inputs_):
        print("tracing -> GeneralAI AC_actor")
        inputs, outputs = inputs_.copy(), {}

        # TODO loop through env specs to get needed storage arrays and numpy_function def
        # metrics = {'rewards_total':tf.float64,'steps':tf.int32,'loss_total':self.compute_dtype,'loss_action':self.compute_dtype,'loss_value':self.compute_dtype,'returns':self.compute_dtype,'advantages':self.compute_dtype,'trans':self.compute_dtype}
        # for k in metrics.keys(): metrics[k] = tf.TensorArray(metrics[k], size=0, dynamic_size=True)
        obs = tf.TensorArray(self.ob_spec['dtype'], size=0, dynamic_size=True)
        actions = tf.TensorArray(self.action_spec['dtype_out'], size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        dones = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True)

        for step in tf.range(self.max_steps):
            # inputs['obs'] = tf.concat([inputs['obs'], tf.cast(inputs['rewards'],self.obs_dtype), tf.cast(inputs['dones'],self.obs_dtype)], axis=1)
            obs = obs.write(step, inputs['obs'][-1])
            returns = returns.write(step, [0.0])

            action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)
            action = action_dist.sample()
            actions = actions.write(step, action[-1])

            if self.force_cont_action and self.action_spec['is_discrete']: action = util.discretize(action, self.action_spec['min'], self.action_spec['max'])
            action = tf.cast(action, self.action_spec['dtype'])
            action = tf.squeeze(action)
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_step, [action], (self.ob_spec['dtype'], tf.float64, tf.bool))

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)
            if inputs['dones'][-1][0]: break

        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = obs.stack(), actions.stack(), rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    @tf.function
    def AC_learner(self, inputs, training=True):
        print("tracing -> GeneralAI AC_learner")

        action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)

        values = self.value(inputs)
        values = tf.squeeze(values, axis=-1)
        
        returns = tf.squeeze(inputs['returns'], axis=-1)
        returns = tf.cast(returns, self.compute_dtype)

        advantages = returns - values
        # advantages = inputs['rewards'] - values
        
        loss = {}
        loss['action'] = self.loss_AC(action_dist, inputs['actions'], advantages)
        # loss['action'] = self.loss_AC(action_dist, inputs['actions'], inputs['rewards'])
        # loss['action'] = self.loss_AC(action_dist, inputs['actions'], returns)
        loss['value'] = self.loss_diff(advantages)
        loss['total'] = loss['action'] + loss['value']

        loss['advantages'] = advantages
        return loss

    @tf.function
    def AC_run(self):
        print("tracing -> GeneralAI AC_run")
        for episode in tf.range(self.max_episodes):
            self.action.reset_states(); self.value.reset_states()
            inputs = {}
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_reset, [], (self.ob_spec['dtype'], tf.float64, tf.bool))
            # TODO how unlimited length episodes without sacrificing returns signal?
            while not inputs['dones'][-1][0]:
                outputs, inputs = self.AC_actor(inputs)
                with tf.GradientTape() as tape:
                    loss = self.AC_learner(outputs)
                gradients = tape.gradient(loss['total'], self.action.trainable_variables + self.value.trainable_variables)
                # isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(gradients[0]), tf.math.is_inf(gradients[0])))
                # if isinfnan > 0: tf.print('\ngradients', gradients[0]); break
                self._optimizer.apply_gradients(zip(gradients, self.action.trainable_variables + self.value.trainable_variables))

                metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                    tf.math.reduce_mean(loss['total']), tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(loss['value']),
                    tf.math.reduce_mean(outputs['returns']), tf.math.reduce_mean(loss['advantages']), False, False, False, False]
                tf.numpy_function(self.metrics_update, metrics, ())



    @tf.function
    def TEST_actor(self, inputs_):
        print("tracing -> GeneralAI TEST_actor")
        inputs, outputs = inputs_.copy(), {}

        # obs = tf.TensorArray(self.ob_spec['dtype'], size=0, dynamic_size=True)
        # latents = tf.TensorArray(self.latent_dtype, size=0, dynamic_size=True)
        latents_next = tf.TensorArray(self.latent_dtype, size=0, dynamic_size=True)
        # actions = tf.TensorArray(self.action_spec['dtype_out'], size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        # dones = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        # returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        targets = tf.TensorArray(self.ob_spec['dtype'], size=0, dynamic_size=True)

        inputs_rep = {'obs':self.latent_zero, 'actions':self.action_zero_out}
        for step in tf.range(self.max_steps):
            # obs = obs.write(step, inputs['obs'][-1])
            # returns = returns.write(step, [0.0])

            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs_rep['obs'] = rep_dist.sample()
            # latents = latents.write(step, inputs_rep['obs'][-1])

            trans_logits = self.trans(inputs_rep); trans_dist = self.trans.dist(trans_logits)
            inputs_rep['obs'] = trans_dist.sample()
            latents_next = latents_next.write(step, inputs_rep['obs'][-1])

            action_logits = self.action(inputs_rep); action_dist = self.action.dist(action_logits)
            action = action_dist.sample()
            inputs_rep['actions'] = action
            # actions = actions.write(step, action[-1])

            if self.force_cont_action and self.action_spec['is_discrete']: action = util.discretize(action, self.action_spec['min'], self.action_spec['max'])
            action = tf.cast(action, self.action_spec['dtype'])
            action = tf.squeeze(action)
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_step, [action], (self.ob_spec['dtype'], tf.float64, tf.bool))

            rewards = rewards.write(step, inputs['rewards'][-1])
            # dones = dones.write(step, inputs['dones'][-1])
            targets = targets.write(step, inputs['obs'][-1])
            # returns_updt = returns.stack()
            # returns_updt = returns_updt + inputs['rewards'][-1]
            # returns = returns.unstack(returns_updt)
            if inputs['dones'][-1][0]: break
        
        # outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = obs.stack(), actions.stack(), rewards.stack(), dones.stack(), returns.stack()
        # outputs['latents'], outputs['latents_next'] = latents.stack(), latents_next.stack()
        outputs['obs'], outputs['rewards'], outputs['targets'] = latents_next.stack(), rewards.stack(), targets.stack()
        return outputs, inputs

    @tf.function
    def TEST_learner(self, inputs_, training=True):
        print("tracing -> GeneralAI TEST_learner")
        inputs = inputs_.copy()
        # inputs = {}
        # obs_next = tf.roll(inputs_['obs'], -1, axis=0)
        # rewards_next = tf.roll(inputs_['rewards'], -1, axis=0)
        # dones_next = tf.roll(inputs_['dones'], -1, axis=0)

        # rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
        # inputs['obs'] = inputs['latents']
        # # inputs['obs'] = rep_dist.sample()
        # # # inputs['obs'] = tf.roll(inputs['latents'], -1, axis=0)

        # trans_logits = self.trans(inputs); trans_dist = self.trans.dist(trans_logits)
        # inputs['obs'] = inputs_['latents_next']
        # inputs['obs'] = trans_dist.sample()

        # rwd_logits = self.rwd(inputs); rwd_dist = self.rwd.dist(rwd_logits) # TODO should it be predicting the rewards_next, dones_next instead?
        # rewards_next = rwd_dist.sample()
        # done_logits = self.done(inputs); done_dist = self.done.dist(done_logits)

        action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)
        # actions_next = tf.roll(inputs['actions'], -1, axis=0)
        # inputs['actions'] = action_dist.sample()

        # values = self.value(inputs)
        # values = tf.squeeze(values, axis=-1)

        # returns = tf.squeeze(inputs['returns'], axis=-1)
        # returns = tf.cast(returns, self.compute_dtype)
        # advantages = returns - values

        loss = {}
        # loss['action'] = self.loss_AC(action_dist, inputs['actions'], advantages)
        # loss['value'] = self.loss_diff(advantages)
        # loss['total'] = loss['action'] + loss['value']

        # loss['rep'] = self.loss_bound(rep_dist, inputs_['obs'])
        # loss['trans'] = self.loss_likelihood(trans_dist, obs_next)
        loss['action'] = self.loss_likelihood(action_dist, inputs['targets'])
        # loss['total'] = loss['rep'] + loss['trans'] + loss['action']
        loss['total'] = loss['action']

        # loss['rwd'] = self.loss_likelihood(rwd_dist, rewards_next)
        # loss['done'] = self.loss_likelihood(done_dist, dones_next)
        # loss['total'] = loss['rep'] + loss['action'] + loss['value'] + loss['trans'] + loss['rwd'] + loss['done']
        # loss['total'] = loss['action'] + loss['value'] + loss['rwd'] + loss['done']
        # loss['total'] = loss['rep'] + loss['action'] + loss['value']

        # loss['advantages'] = advantages
        return loss

    @tf.function
    def TEST_imagine(self, inputs_):
        print("tracing -> GeneralAI TEST_imagine")
        inputs, outputs = inputs_.copy(), {}

        obs = tf.TensorArray(self.latent_dtype, size=0, dynamic_size=True)
        actions = tf.TensorArray(self.action_spec['dtype_out'], size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        dones = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True)

        for step in tf.range(self.max_steps):
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs['actions'], self.action_invar), (inputs['dones'], self.action_invar), (inputs['rewards'], self.action_invar)])
            obs = obs.write(step, inputs['obs'][-1])
            returns = returns.write(step, [0.0])

            action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)
            # inputs['actions'] = action_dist.sample()
            # actions = actions.write(step, inputs['actions'][-1])
            action = action_dist.sample()
            actions = actions.write(step, action[-1])
            inputs['actions'] = action

            trans_logits = self.trans(inputs); trans_dist = self.trans.dist(trans_logits)
            inputs['obs'] = trans_dist.sample()
            rwd_logits = self.rwd(inputs); rwd_dist = self.rwd.dist(rwd_logits)
            done_logits = self.done(inputs); done_dist = self.done.dist(done_logits)
            inputs['rewards'], inputs['dones'] = tf.cast(rwd_dist.sample(), tf.float64), tf.cast(done_dist.sample(), tf.bool)

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)
            if inputs['dones'][-1][0]: break

        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = obs.stack(), actions.stack(), rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    @tf.function
    def TEST_real(self, inputs_):
        print("tracing -> GeneralAI TEST_real")
        inputs, outputs = inputs_.copy(), {}

        rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
        outputs['obs'] = rep_dist.sample()

        action_logits = self.action(outputs); action_dist = self.action.dist(action_logits)
        outputs['actions'] = action_dist.sample()

        trans_logits = self.trans(outputs); trans_dist = self.trans.dist(trans_logits)
        outputs['obs'] = trans_dist.sample()

        gen_logits = self.gen(outputs); gen_dist = self.gen.dist(gen_logits)
        # obs_gen = gen_dist.sample()
        # if self.force_cont_ob and self.ob_spec['is_discrete']: obs_gen = util.discretize(obs_gen, self.ob_spec['min'], self.ob_spec['max'])
        # outputs['obs_gen'] = gen_dist.sample()
        rwd_logits = self.rwd(outputs); rwd_dist = self.rwd.dist(rwd_logits)
        done_logits = self.done(outputs); done_dist = self.done.dist(done_logits)

        action = outputs['actions']
        # action = tf.stop_gradient(outputs['actions']) # TODO stop gradient?
        if self.force_cont_action and self.action_spec['is_discrete']: action = util.discretize(action, self.action_spec['min'], self.action_spec['max'])
        action = tf.cast(action, self.action_spec['dtype'])
        action = tf.squeeze(action)
        inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_step, [action], (self.ob_spec['dtype'], tf.float64, tf.bool))
        
        loss = {}
        loss['restruct'] = self.loss_likelihood(gen_dist, inputs['obs'])
        loss['rewards'] = self.loss_likelihood(rwd_dist, inputs['rewards'])
        loss['dones'] = self.loss_likelihood(done_dist, inputs['dones'])
        loss['total'] = loss['restruct'] + loss['rewards'] + loss['dones']
        return outputs, inputs, loss

    @tf.function
    def TEST_run(self):
        print("tracing -> GeneralAI TEST_run")
        for episode in tf.range(self.max_episodes):
            # self.rep.reset_states(); self.trans.reset_states(); self.action.reset_states(); self.value.reset_states()
            inputs = {}
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_reset, [], (self.ob_spec['dtype'], tf.float64, tf.bool))
            while not inputs['dones'][-1][0]:
                # outputs, inputs = self.TEST_actor(inputs)
                with tf.GradientTape() as tape:
                    # outputs, inputs = self.TEST_actor(inputs)
                    # loss = self.TEST_learner(outputs)
                    outputs, inputs, loss = self.TEST_real(inputs)
                # gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables + self.done.trainable_variables)
                # self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables + self.done.trainable_variables))
                gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.trans.trainable_variables + self.gen.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.trans.trainable_variables + self.gen.trainable_variables))

                inputs_imag = {}
                inputs_imag['obs'], inputs_imag['rewards'], inputs_imag['dones'] = outputs['obs'], tf.constant([[0]],tf.float64), tf.constant([[False]],tf.bool)
                inputs_imag['actions'] = self.action_zero_out
                while not inputs_imag['dones'][-1][0]:
                    # tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs_imag['dones'], self.action_invar)])
                    outputs_imag, inputs_imag = self.TEST_imagine(inputs_imag)
                    with tf.GradientTape() as tape:
                        loss_imag = self.AC_learner(outputs_imag)
                    gradients = tape.gradient(loss_imag['total'], self.action.trainable_variables + self.value.trainable_variables)
                    self._optimizer.apply_gradients(zip(gradients, self.action.trainable_variables + self.value.trainable_variables))

                # metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                metrics = [episode, tf.math.reduce_sum(inputs['rewards']), inputs['rewards'][-1][0], tf.shape(inputs['rewards'])[0],
                    tf.math.reduce_mean(loss['total']), False, False, False, False,
                    # tf.math.reduce_mean(loss['total']), tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(loss['value']),
                    # tf.math.reduce_mean(outputs['returns']), tf.math.reduce_mean(loss['advantages']),
                    False, False, False, False
                    # tf.math.reduce_mean(loss['rep']), False, False, False
                    # tf.math.reduce_mean(loss['rep']), tf.math.reduce_mean(loss['trans']), False, False
                    # tf.math.reduce_mean(loss['rwd']), tf.math.reduce_mean(loss['done'])
                ]
                tf.numpy_function(self.metrics_update, metrics, ())




def params(): pass
load_model = False
max_episodes = 500
learn_rate = 1e-5
entropy_contrib = 1e-8
returns_disc = 1.0
force_cont_ob, force_cont_action = True, True
latent_size = 64
latent_cat = True
latent_size_mem_multi = 1

device_type = 'GPU' # use GPU for large networks or big data
device_type = 'CPU'

machine, device = 'dev', 0

env_name, max_steps, render, env = 'CartPole', 256, False, gym.make('CartPole-v0'); env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, render, env = 'CartPole', 512, False, gym.make('CartPole-v1'); env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, render, env = 'LunarLand', 1024, False, gym.make('LunarLander-v2')
# env_name, max_steps, render, env = 'LunarLandCont', 1024, False, gym.make('LunarLanderContinuous-v2')
# import envs_local.random as env_; env_name, max_steps, render, env = 'TestRnd', 128, False, env_.RandomEnv()
# import envs_local.data as env_; env_name, max_steps, render, env = 'DataShkspr', 16, True, env_.DataEnv('shkspr')
# import envs_local.data as env_; env_name, max_steps, render, env = 'DataMnist', 128, False, env_.DataEnv('mnist')
# import envs_local.bipedal_walker as env_; env_name, max_steps, render, env = 'BipedalWalker', 128, False, env_.BipedalWalker()
# trader, trader_env, trader_speed = False, 3, 180.0
# import gym_trader; env_name, max_steps, render, trader, env = 'Trader2', 128, False, True, gym.make('Trader-v0', agent_id=device, env=trader_env, speed=trader_speed)

# import envs_local.async_wrapper as env_async_wrapper; env_name, env = env_name+'-Asyn', env_async_wrapper.AsyncWrapperEnv(env)

# TODO try TD error with batch one
# arch = 'TEST' # testing architechures
# arch = 'DNN' # basic Deep Neural Network, likelyhood loss
# arch = 'TRANS' # learned Transition dynamics, autoregressive likelyhood loss
arch = 'AC' # basic Actor Critic, actor/critic loss
# arch = 'DREAM' # full World Model w/imagination (DeepMind Dreamer)
# arch = 'MU' # Dreamer/planner w/imagination (DeepMind MuZero)

if __name__ == '__main__':
    # TODO add keyboard control so can stop

    ## manage multiprocessing
    # # setup ctrl,data,param sharing
    # # start agents (real+dreamers)
    # agent = Agent(model)
    # # agent_process = mp.Process(target=agent.vivify, name='AGENT', args=(lock_print, process_ctrl, weights_shared))
    # # agent_process.start()
    # # quit on keyboard (space = save, esc = no save)
    # process_ctrl.value = 0
    # agent_process.join()

    with tf.device("/device:{}:{}".format(device_type,device)):
        model = GeneralAI(arch, env, render, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, force_cont_ob, force_cont_action, latent_size, latent_cat, memory_size=max_steps*latent_size_mem_multi)
        name = "gym-{}-{}-{}".format(arch, env_name, ('Lcat' if latent_cat else 'Lcon'))


        ## load models
        model_files, name_arch = {}, ""
        for net in model.layers:
            model_name = "{}-{}-a{}".format(net.net_arch, machine, device)
            model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
            model_files[net.name] = model_file
            if load_model and tf.io.gfile.exists(model_file):
                net.load_weights(model_file, by_name=True, skip_mismatch=True)
                print("LOADED {} weights from {}".format(net.name, model_file)); loaded_model = True
            name_arch += "   {}{}".format(net.net_arch, 'load' if loaded_model else 'new')
        # print(model.call.pretty_printed_concrete_signatures()); quit(0)
        # model.summary(); quit(0)


        ## run
        t1_start = time.perf_counter_ns()
        if arch=='TEST': model.TEST_run()
        if arch=='TRANS': model.TRANS_run()
        if arch=='AC': model.AC_run()
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds


        ## metrics
        metrics, metrics_loss = model.metrics_main, model.metrics_loss
        for loss_group in metrics_loss:
            for k in loss_group.keys():
                for j in range(len(loss_group[k])): loss_group[k][j] = np.mean(loss_group[k][j])
        # TODO np.mean, reduce size if above 100,000-200,000 episodes

        name = "{}-{}-a{}-{}".format(name, machine, device, time.strftime("%Y_%m_%d-%H-%M"))
        total_steps = np.sum(metrics['steps'])
        step_time = total_time/total_steps
        title = "{} {}\ntime:{}    steps:{}    t/s:{:.8f}     |     lr:{}    dis:{}    en:{}".format(name, name_arch, util.print_time(total_time), total_steps, step_time, learn_rate, returns_disc, entropy_contrib); print(title)

        import matplotlib as mpl
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'lightblue', 'green', 'lime', 'red', 'lavender', 'turquoise', 'cyan', 'magenta', 'salmon', 'yellow', 'gold', 'black', 'brown', 'purple', 'pink', 'orange', 'teal', 'coral', 'darkgreen', 'tan'])
        plt.figure(num=name, figsize=(34, 16), tight_layout=True)
        xrng, i, vplts = np.arange(0, max_episodes, 1), 0, 3 + len(metrics_loss)

        rows = 2; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
        metric_name = 'rewards_total'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        metric_name = 'rewards_final'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left'); plt.title(title)

        for loss_group in metrics_loss:
            rows = 1; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
            for metric_name, metric in loss_group.items():
                metric = np.asarray(metric, np.float64); plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
            plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left')
        
        rows = 1; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
        metric_name = 'steps'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left')

        plt.show()


        ## save models
        for net in model.layers:
            model_file = model_files[net.name]
            net.save_weights(model_file)
            print("SAVED {} weights to {}".format(net.name, model_file))
