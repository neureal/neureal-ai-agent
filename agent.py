import time, os, talib
import multiprocessing as mp
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
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
# TODO try out MuZero-ish architecture (DREAM)
# TODO add imagination by looping through TransNet seperately from looping through env.step

# TODO add in generic gym stuff from model_util
# TODO try embedding all the net imputs to latent_size and use # out = tf.math.accumulate_n(out_accu)
# TODO combine all the diff architectures into one and use # if arch == 'NAME': or # if arch in ('TRANS', 'DREAM'):
# TODO wrap env in seperate process and make run async with random NOOP skips to test latency learning/timing
# TODO put actor in seperate process so can run async
# TODO add GenNet and DisNet for World Model (WM)
# TODO use attention (transformer decoder) layers instead of LSTM

# TODO how to incorporate ARS random policy search?
# TODO try out the 'lottery ticket hypothosis' pruning during training
# TODO use numba to make things faster on CPU


class RepNet(tf.keras.layers.Layer):
    def __init__(self, latent_size, categorical):
        super(RepNet, self).__init__()
        self.categorical, event_shape = categorical, (latent_size,)

        if categorical: num_components = 256; params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        else: num_components = latent_size; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)
        
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)

        self.net_blocks, self.net_LSTM, inp, mid, evo = 1, False, 256, 256, 32
        self.net_arch = "RN[inD{}-{:02d}{}D{}-cmp{}-lat{}]".format(inp, self.net_blocks, ('LS+' if self.net_LSTM else ''), mid, num_components, latent_size)

        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_LSTM: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))

        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

    def _net(self, inputs):
        out = self.layer_dense_in(inputs)
        for i in range(self.net_blocks):
            if self.net_LSTM: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_logits_out(out)
        return out

    def reset_states(self):
        if self.net_LSTM:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
            # if k == 'rewards':
            #     out = tf.cast(v, self.compute_dtype)
            #     out_accu.append(out)
            # if k == 'dones':
            #     out = tf.cast(v, self.compute_dtype)
            #     out_accu.append(out)
        out = tf.concat(out_accu, 1)
        out = self._net(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('rep net out:', out)
        return out

    def loss(self, dist, targets):
        targets = tf.cast(targets, dist.dtype)
        loss = dist.log_prob(targets)
        if not self.categorical: loss = loss - self.dist_prior.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('rep net loss:', loss)
        return loss


# transition dynamics within latent space
class TransNet(tf.keras.layers.Layer):
    def __init__(self, latent_size, categorical):
        super(TransNet, self).__init__()
        self.categorical, event_shape = categorical, (latent_size,)

        if categorical: num_components = 256; params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        else: num_components = latent_size * 8; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)

        self.net_blocks, self.net_LSTM, inp, mid, evo = 1, True, 256, 256, 32
        self.net_arch = "TN[inD{}-{:02d}{}D{}-cmp{}-lat{}]".format(inp, self.net_blocks, ('LS+' if self.net_LSTM else ''), mid, num_components, latent_size)

        self.layer_cond_dense_in = tf.keras.layers.Dense(64, activation=util.EvoNormS0(8), use_bias=False, name='cond_dense_in')
        self.layer_cond_dense_latent = tf.keras.layers.Dense(latent_size, name='cond_dense_latent')

        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_LSTM: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))

        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

    def _net(self, inputs):
        out = self.layer_dense_in(inputs)
        for i in range(self.net_blocks):
            if self.net_LSTM: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_logits_out(out)
        return out

    def reset_states(self):
        if self.net_LSTM:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'actions':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_cond_dense_in(out)
                out = self.layer_cond_dense_latent(out)
                out_accu.append(out)
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
        out = tf.math.accumulate_n(out_accu)
        # out = tf.concat(out_accu, 1)
        out = self._net(out)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('trans net out:', out)
        return out

    def loss(self, dist, targets):
        targets = tf.cast(targets, dist.dtype)
        loss = -dist.log_prob(targets)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('trans net loss:', loss)
        return loss


class RewardNet(tf.keras.layers.Layer):
    def __init__(self):
        super(RewardNet, self).__init__()
        num_components, event_shape = 16, (1,); params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)
        inp, evo = 256, 32; self.net_arch = "RWD[inD{}-cmp{}]".format(inp, num_components)
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        # self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
    @tf.function
    def call(self, inputs, training=None):
        out = self.layer_dense_in(inputs['obs'])
        out = self.layer_dense_logits_out(out)
        return out
    def loss(self, dist, targets):
        targets = tf.cast(targets, dist.dtype)
        loss = -dist.log_prob(targets)
        return loss

class DoneNet(tf.keras.layers.Layer):
    def __init__(self):
        super(DoneNet, self).__init__()
        num_components, event_shape = 16, (1,); params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)
        inp, evo = 256, 32; self.net_arch = "DON[inD{}-cmp{}]".format(inp, num_components)
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        # self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
    @tf.function
    def call(self, inputs, training=None):
        out = self.layer_dense_in(inputs['obs'])
        out = self.layer_dense_logits_out(out)
        return out
    def loss(self, dist, targets):
        targets = tf.cast(targets, dist.dtype)
        loss = -dist.log_prob(targets)
        return loss


class ActionNet(tf.keras.layers.Layer):
    def __init__(self, env, categorical, entropy_contrib):
        super(ActionNet, self).__init__()
        self.entropy_contrib, self.categorical, self.is_discrete = tf.constant(entropy_contrib, self.compute_dtype), categorical, False

        if isinstance(env.action_space, gym.spaces.Discrete): num_components, event_shape, self.is_discrete = env.action_space.n, (1,), True
        elif isinstance(env.action_space, gym.spaces.Box): event_shape = list(env.action_space.shape); num_components = np.prod(event_shape).item()

        if categorical: params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        else: num_components *= 4; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)

        self.net_blocks, self.net_LSTM, inp, mid, evo = 1, False, 256, 256, 32
        self.net_arch = "AN[inD{}-{:02d}{}D{}-cmp{}]".format(inp, self.net_blocks, ('LS+' if self.net_LSTM else ''), mid, num_components)

        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_LSTM: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
        
        self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')

    def reset_states(self):
        if self.net_LSTM:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
            if k == 'obs_pred':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
        out = tf.concat(out_accu, 1)
        out = self.layer_dense_in(out)
        for i in range(self.net_blocks):
            if self.net_LSTM: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_logits_out(out)
        
        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('action net out:', out)
        return out

    def loss(self, dist, targets, advantages):
        targets = tf.cast(targets, dist.dtype)
        loss = -dist.log_prob(targets)
        loss = loss * advantages # * 1e-2
        # if self.categorical:
        #     entropy = dist.entropy()
        #     loss = loss - entropy * self.entropy_contrib # "Soft Actor Critic" = try increase entropy

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)))
        if isinfnan > 0: tf.print('action net loss:', loss)
        return loss


class ValueNet(tf.keras.layers.Layer):
    def __init__(self, env):
        super(ValueNet, self).__init__()
        self.net_blocks, self.net_LSTM, inp, mid, evo = 1, False, 128, 128, 32
        self.net_arch = "VN[inD{}-{:02d}{}D{}]".format(inp, self.net_blocks, ('LS+' if self.net_LSTM else ''), mid)

        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_LSTM: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
        self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')

    def reset_states(self):
        if self.net_LSTM:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
            if k == 'obs_pred':
                out = tf.cast(v, self.compute_dtype)
                out = self.layer_flatten(out)
                out_accu.append(out)
            # if k == 'actions':
            #     out = tf.cast(v, self.compute_dtype)
            #     out_accu.append(out)
        out = tf.concat(out_accu, 1)
        out = self.layer_dense_in(out)
        for i in range(self.net_blocks):
            if self.net_LSTM: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        out = self.layer_dense_out(out)
        return out

    def loss(self, advantages):
        loss = tf.where(tf.math.less(advantages, 0.0), tf.math.negative(advantages), advantages) # MAE
        return loss


class GeneralAI(tf.keras.Model):
    def __init__(self, arch, env, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, action_cat, latent_size, latent_cat, memory_size):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.arch, self.env, self.max_episodes, self.max_steps, self.returns_disc = arch, env, tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(returns_disc, tf.float64)
        self.float_maxroot = tf.constant(tf.math.sqrt(compute_dtype.max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)

        # TODO add generic env specs here
        # self.obs_dtypes, self.obs_spec, self.obs_sample = util.gym_get_spec(env.observation_space)
        # self.action_dtypes, self.action_spec, self.action_sample = util.gym_get_spec(env.action_space)
        self.obs_dtype = tf.dtypes.as_dtype(env.observation_space.dtype)
        self.obs_zero = tf.constant(0, self.obs_dtype, shape=[1]+list(env.observation_space.shape))

        self.action_dtype = tf.dtypes.as_dtype(env.action_space.dtype)
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_shape = [1,]
            self.action_min = tf.constant(0, compute_dtype)
            self.action_max = tf.constant(env.action_space.n-1, compute_dtype)
        elif isinstance(env.action_space, gym.spaces.Box):
            action_shape = list(env.action_space.shape)
            self.action_min = tf.constant(env.action_space.low[...,0], compute_dtype)
            self.action_max = tf.constant(env.action_space.high[...,0], compute_dtype)
        self.action_zero = tf.constant(0, self.action_dtype, shape=[1]+action_shape)
        self.action_out_dtype = tf.int32 if action_cat else compute_dtype

        # if arch in ('TRANS'): latent_size = env.observation_space.shape[0]
        self.latent_dtype = tf.int32 if latent_cat else compute_dtype
        self.latent_zero, self.latent_invar = tf.constant(0, self.latent_dtype, shape=(1,latent_size)), tf.TensorShape([None,latent_size])


        inputs = {'actions':self.action_zero, 'obs':self.obs_zero, 'rewards':tf.constant([[0]],tf.float64), 'dones':tf.constant([[False]],tf.bool)}
        # self.obs_zero = tf.concat([inputs['obs'], tf.cast(inputs['rewards'],self.obs_dtype), tf.cast(inputs['dones'],self.obs_dtype)], axis=1)
        # inputs = {'obs':self.obs_zero}
        if arch in ('DREAM'):
            self.rep = RepNet(latent_size, latent_cat); outputs = self.rep(inputs)

        # if arch in ('TRANS'): inputs['obs_pred'] = self.latent_zero
        if arch in ('DREAM'):
            inputs['obs'] = self.latent_zero
            self.rwd = RewardNet(); outputs = self.rwd(inputs)
            self.done = DoneNet(); outputs = self.done(inputs)
        self.action = ActionNet(env, action_cat, entropy_contrib); outputs = self.action(inputs)
        self.value = ValueNet(env); outputs = self.value(inputs)

        if arch in ('DREAM'):
            self.trans = TransNet(latent_size, latent_cat); outputs = self.trans(inputs)

        self.reset_states()
        self.discounted_sum = tf.Variable(0, dtype=tf.float64, trainable=False)

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=self.float_eps)


        metrics = {'rewards_total':np.float64,'rewards_final':np.float64,'steps':np.int64}
        metrics_loss = [
            {'loss_total':np.float64},{'loss_action':np.float64,'loss_value':np.float64},
            {'returns':np.float64,'advantages':np.float64}
        ]
        if arch in ('TRANS','DREAM'):
            metrics_loss.append({'loss_rep':np.float64})
            # metrics_loss.append({'loss_rep':np.float64,'loss_trans':np.float64})
        # if arch in ('DREAM'): metrics_loss.append({'loss_rwd':np.float64,'loss_done':np.float64})
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

    def env_reset(self):
        obs, reward, done = self.env.reset(), 0.0, False
        return np.expand_dims(obs,0), np.expand_dims(np.asarray(reward, np.float64),(0,1)), np.expand_dims(np.asarray(done, np.bool),(0,1))
    def env_step(self, action):
        obs, reward, done, _ = self.env.step(action)
        return np.expand_dims(obs,0), np.expand_dims(np.asarray(reward, np.float64),(0,1)), np.expand_dims(np.asarray(done, np.bool),(0,1))

    @tf.function
    def reset_states(self):
        for net in self.layers:
            if hasattr(net, 'reset_states'): net.reset_states()

    def action_discretize(self, action):
        action = tf.math.round(action)
        action = tf.clip_by_value(action, self.action_min, self.action_max)
        return action




    # @tf.function
    # def TRANS_actor(self):
    #     print("tracing -> GeneralAI TRANS_actor")
    #     inputs, outputs = {}, {}
    #     return outputs

    # @tf.function
    # def TRANS_learner(self, inputs_, training=True):
    #     print("tracing -> GeneralAI TRANS_learner")
    #     return loss

    # @tf.function
    # def TRANS_run(self):
    #     print("tracing -> GeneralAI TRANS_run")
    #     return metrics, metrics_loss



    @tf.function
    def AC_actor(self, inputs_):
        print("tracing -> GeneralAI AC_actor")
        inputs, outputs = inputs_.copy(), {}

        # TODO loop through env specs to get needed storage arrays and numpy_function def
        # metrics = {'rewards_total':tf.float64,'steps':tf.int32,'loss_total':self.compute_dtype,'loss_action':self.compute_dtype,'loss_value':self.compute_dtype,'returns':self.compute_dtype,'advantages':self.compute_dtype,'trans':self.compute_dtype}
        # for k in metrics.keys(): metrics[k] = tf.TensorArray(metrics[k], size=0, dynamic_size=True)
        obs = tf.TensorArray(self.obs_dtype, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        dones = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        actions = tf.TensorArray(self.action_out_dtype, size=0, dynamic_size=True)
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True)

        for step in tf.range(self.max_steps):
            # inputs['obs'] = tf.concat([inputs['obs'], tf.cast(inputs['rewards'],self.obs_dtype), tf.cast(inputs['dones'],self.obs_dtype)], axis=1)
            obs = obs.write(step, inputs['obs'][-1])
            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [0.0])

            action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)
            action = action_dist.sample()
            actions = actions.write(step, action[-1])

            if inputs['dones'][-1][0]: break

            if not self.action.categorical and self.action.is_discrete: action = self.action_discretize(action)
            action = tf.cast(action, self.action_dtype)
            action = tf.squeeze(action)
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_step, [action], (self.obs_dtype, tf.float64, tf.bool))

            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)


        outputs['obs'], outputs['rewards'], outputs['dones'], outputs['actions'], outputs['returns'] = obs.stack(), rewards.stack(), dones.stack(), actions.stack(), returns.stack()
        return outputs, inputs

    @tf.function
    def AC_learner(self, inputs, training=True):
        print("tracing -> GeneralAI AC_learner")

        action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)

        values = self.value(inputs)
        values = tf.squeeze(values, axis=-1)
        
        returns = tf.squeeze(inputs['returns'], axis=-1)

        advantages = returns - values
        # advantages = inputs['rewards'] - values
        
        loss = {}
        loss['action'] = self.action.loss(action_dist, inputs['actions'], advantages)
        # loss['action'] = self.action.loss(action_dist, inputs['actions'], inputs['rewards'])
        # loss['action'] = self.action.loss(action_dist, inputs['actions'], returns)
        loss['value'] = self.value.loss(advantages)
        loss['total'] = loss['action'] + loss['value']

        loss['advantages'] = advantages
        return loss

    @tf.function
    def AC_run(self):
        print("tracing -> GeneralAI AC_run")
        for episode in tf.range(self.max_episodes):
            self.action.reset_states(); self.value.reset_states()
            inputs = {}
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_reset, [], (self.obs_dtype, tf.float64, tf.bool))
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
    def DREAM_actor(self, inputs_):
        print("tracing -> GeneralAI DREAM_actor")
        inputs, outputs = inputs_.copy(), {}

        obs = tf.TensorArray(self.obs_dtype, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        dones = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        actions = tf.TensorArray(self.action_out_dtype, size=0, dynamic_size=True)
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        states = tf.TensorArray(self.latent_dtype, size=0, dynamic_size=True)

        inputs_rep = {'obs':self.latent_zero}
        # inputs_rep = {'obs':self.latent_zero, 'obs_pred':self.latent_zero}

        for step in tf.range(self.max_steps):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs_rep['obs'], self.latent_invar)])
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs_rep['obs'], self.latent_invar), (inputs_rep['obs_pred'], self.latent_invar)])
            obs = obs.write(step, inputs['obs'][-1])
            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [0.0])

            rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
            inputs_rep['obs'] = rep_dist.sample()
            states = states.write(step, inputs_rep['obs'][-1])

            # trans_logits = self.trans(inputs_rep); trans_dist = self.trans.dist(trans_logits)
            # inputs_rep['obs_pred'] = trans_dist.sample()

            action_logits = self.action(inputs_rep); action_dist = self.action.dist(action_logits)
            action = action_dist.sample()
            actions = actions.write(step, action[-1])
            
            if inputs['dones'][-1][0]: break

            if not self.action.categorical and self.action.is_discrete: action = self.action_discretize(action)
            action = tf.cast(action, self.action_dtype)
            action = tf.squeeze(action)
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_step, [action], (self.obs_dtype, tf.float64, tf.bool))

            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)
        
        outputs['obs'], outputs['rewards'], outputs['dones'], outputs['actions'], outputs['returns'], outputs['states'] = obs.stack(), rewards.stack(), dones.stack(), actions.stack(), returns.stack(), states.stack()
        return outputs, inputs

    @tf.function
    def DREAM_learner(self, inputs_, training=True):
        print("tracing -> GeneralAI DREAM_learner")
        inputs = inputs_.copy()
        # rewards_next = tf.roll(inputs['rewards'], -1, axis=0)
        # dones_next = tf.roll(inputs['dones'], -1, axis=0)

        rep_logits = self.rep(inputs); rep_dist = self.rep.dist(rep_logits)
        inputs['obs'] = inputs['states']
        # inputs['obs'] = rep_dist.sample()
        # # inputs['obs'] = tf.roll(inputs['states'], -1, axis=0)

        # trans_logits = self.trans(inputs); trans_dist = self.trans.dist(trans_logits)
        # inputs['obs_pred'] = trans_dist.sample()
        # # obs_next = tf.roll(inputs['obs'], -1, axis=0)

        # rwd_logits = self.rwd(inputs); rwd_dist = self.rwd.dist(rwd_logits) # TODO should it be predicting the rewards_next, dones_next instead?
        # rewards_next = rwd_dist.sample()
        # done_logits = self.done(inputs); done_dist = self.done.dist(done_logits)

        action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)
        # actions_next = tf.roll(inputs['actions'], -1, axis=0)
        # inputs['actions'] = action_dist.sample()

        values = self.value(inputs)
        values = tf.squeeze(values, axis=-1)

        returns = tf.squeeze(inputs['returns'], axis=-1)
        advantages = returns - values

        loss = {}
        loss['rep'] = self.rep.loss(rep_dist, inputs['obs'])
        loss['action'] = self.action.loss(action_dist, inputs['actions'], advantages)
        # loss['action'] = self.action.loss(action_dist, actions_next, advantages)
        loss['value'] = self.value.loss(advantages)
        # loss['trans'] = self.trans.loss(trans_dist, obs_next)
        # loss['rwd'] = self.rwd.loss(rwd_dist, rewards_next)
        # loss['done'] = self.done.loss(done_dist, dones_next)
        # loss['total'] = loss['rep'] + loss['action'] + loss['value'] + loss['trans'] + loss['rwd'] + loss['done']
        # loss['total'] = loss['action'] + loss['value'] + loss['rwd'] + loss['done']
        loss['total'] = loss['rep'] + loss['action'] + loss['value']
        # loss['total'] = loss['action'] + loss['value']

        loss['advantages'] = advantages
        return loss

    # @tf.function
    # def DREAM_imagine_actor(self):
    #     print("tracing -> GeneralAI DREAM_imagine_actor")

    # @tf.function
    # def DREAM_imagine_learner(self, inputs_, training=True):
    #     print("tracing -> GeneralAI DREAM_imagine_learner")

    @tf.function
    def DREAM_run(self):
        print("tracing -> GeneralAI DREAM_run")
        for episode in tf.range(self.max_episodes):
            # self.trans.reset_states()
            self.rep.reset_states(); self.action.reset_states(); self.value.reset_states()
            inputs = {}
            inputs['obs'], inputs['rewards'], inputs['dones'] = tf.numpy_function(self.env_reset, [], (self.obs_dtype, tf.float64, tf.bool))
            while not inputs['dones'][-1][0]:
                outputs, inputs = self.DREAM_actor(inputs)

                with tf.GradientTape() as tape:
                    loss = self.DREAM_learner(outputs)
                # gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables + self.done.trainable_variables)
                # self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables + self.trans.trainable_variables + self.rwd.trainable_variables + self.done.trainable_variables))
                gradients = tape.gradient(loss['total'], self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, self.rep.trainable_variables + self.action.trainable_variables + self.value.trainable_variables))

                metrics = [episode, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                    tf.math.reduce_mean(loss['total']), tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(loss['value']),
                    tf.math.reduce_mean(outputs['returns']), tf.math.reduce_mean(loss['advantages']),
                    tf.math.reduce_mean(loss['rep']),
                    # tf.math.reduce_mean(loss['rep']), tf.math.reduce_mean(loss['trans'])
                    # tf.math.reduce_mean(loss['rwd']), tf.math.reduce_mean(loss['done'])
                    False, False, False
                ]
                tf.numpy_function(self.metrics_update, metrics, ())




def params(): pass
max_episodes = 1000
learn_rate = 1e-5
entropy_contrib = 1e-8
returns_disc = 0.99
action_cat = True
latent_size = 8
latent_cat = False

mem_size_multi = 2

machine, device = 'dev', 0

env_name, max_steps, env = 'CartPole', 16, gym.make('CartPole-v0'); env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, env = 'LunarLand', 1024, gym.make('LunarLander-v2')
# env_name, max_steps, env = 'LunarLandCont', 1024, gym.make('LunarLanderContinuous-v2')
# import envs_local.random as env_; env_name, max_steps, env = 'TestRnd', 128, env_.RandomEnv()
# import envs_local.data as env_; env_name, max_steps, env = 'DataShkspr', 128, env_.DataEnv('shkspr')
# import envs_local.data as env_; env_name, max_steps, env = 'DataMnist', 128, env_.DataEnv('mnist')
# import envs_local.bipedal_walker as env_; env_name, max_steps, env = 'BipedalWalker', 128, env_.BipedalWalker()
# import gym_trader; env_name, max_steps, trader, env = 'Trader2', 128, True, gym.make('Trader-v0', agent_id=device, env=trader_env, speed=trader_speed)
# trader, trader_env, trader_speed = False, 3, 180.0

# import envs_local.async_wrapper as env_async_wrapper; env_name, env = env_name+'-Asyn', env_async_wrapper.AsyncWrapperEnv(env)

# arch = 'DNN' # basic Deep Neural Network, likelyhood loss
# arch = 'TRANS' # learned Transition dynamics, autoregressive likelyhood loss
# arch = 'AC' # basic Actor Critic, actor/critic loss
# arch = 'MU' # Dreamer/planner w/imagination (DeepMind MuZero)
arch = 'DREAM' # full World Model w/imagination (DeepMind Dreamer)

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


    # with tf.device('/device:GPU:0'): # use GPU for large networks or big data
    with tf.device('/device:CPU:0'):
        model = GeneralAI(arch, env, max_episodes=max_episodes, max_steps=max_steps, learn_rate=learn_rate, entropy_contrib=entropy_contrib, returns_disc=returns_disc, action_cat=action_cat, latent_size=latent_size, latent_cat=latent_cat, memory_size=max_steps*mem_size_multi)
        name = "gym-{}-{}-{}-{}".format(arch, env_name, ('Acat' if action_cat else 'Acon'), ('Lcat' if latent_cat else 'Lcon'))


        # # TODO load models, load each net seperately
        # self.net_arch = "{}-{}-{}".format(self.trans.net_arch, self.action.net_arch, self.value.net_arch)
        model_name = "{}-{}-{}-a{}".format(name, model.action.net_arch, machine, device)
        model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
        # if tf.io.gfile.exists(model_file):
        #     model.load_weights(model_file, by_name=True, skip_mismatch=True)
        #     print("LOADED model weights from {}".format(model_file)); loaded_model = True
        # # print(model.call.pretty_printed_concrete_signatures()); quit(0)
        # # model.summary(); quit(0)

        ## run
        t1_start = time.perf_counter_ns()
        # if arch=='TRANS': model.TRANS_run()
        if arch=='AC': model.AC_run()
        if arch=='DREAM': model.DREAM_run()
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds

        metrics, metrics_loss = model.metrics_main, model.metrics_loss
        for loss_group in metrics_loss:
            for k in loss_group.keys():
                for j in range(len(loss_group[k])): loss_group[k][j] = np.mean(loss_group[k][j])
        # TODO np.mean, reduce size if above 100,000-200,000 episodes

        # # TODO save models
        # model.save_weights(model_file)
        
        ## metrics
        name, name_arch = "{}-{}-a{}-{}".format(name, machine, device, time.strftime("%Y_%m_%d-%H-%M")), ""
        for net in model.layers: name_arch += "   "+net.net_arch
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
