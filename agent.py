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

# TODO add imagination by looping through TransNet seperately from env.step
# TODO test conditioning with action
# TODO add RepNet, GenNet and DisNet
# TODO use numba to make things faster on CPU


# transition dynamics within latent space
class TransNet(tf.keras.layers.Layer):
    def __init__(self, latent_size, categorical):
        super(TransNet, self).__init__()
        self.categorical, event_shape = categorical, (latent_size,)

        if categorical: num_components = 256; params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        else: num_components = latent_size * 8; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)

        self.net_blocks, self.net_LSTM, inp, mid, evo = 1, True, 256, 256, 32
        self.net_arch = "TN[inD{}-{:02d}{}D{}-cmp{}-lat{}]".format(inp, self.net_blocks, ('LS+' if self.net_LSTM else ''), mid, num_components, latent_size)

        # self.layer_cond_dense_in = tf.keras.layers.Dense(64, activation=util.EvoNormS0(8), use_bias=False, name='cond_dense_in')
        # self.layer_cond_dense_latent_out = tf.keras.layers.Dense(latent_size, name='cond_dense_latent_out')

        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_LSTM: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))

        if categorical: self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        else:
            self.split_cats, loc_scale_size_each = tf.constant(num_components, tf.int32), int((params_size-num_components)/2)
            self.layer_cont_cats, self.layer_cont_loc, self.layer_cont_scale = tf.keras.layers.Dense(num_components), tf.keras.layers.Dense(loc_scale_size_each), tf.keras.layers.Dense(loc_scale_size_each)

    # def _net_cond(self, inputs):
    #     out = self.layer_cond_dense_in(inputs)
    #     out = self.layer_cond_dense_latent_out(out)
    #     return out
    def _net(self, inputs):
        out = self.layer_dense_in(inputs)
        for i in range(self.net_blocks):
            if self.net_LSTM: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        if self.categorical: out = self.layer_dense_logits_out(out)
        else: out = util.combine_logits(out, self.layer_cont_cats, self.layer_cont_loc, self.layer_cont_scale, self.split_cats)
        return out

    def reset_states(self):
        if self.net_LSTM:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            # if k == 'action':
            #     out = tf.cast(v, self.compute_dtype)
            #     out = self._net_cond(out)
            #     out_accu.append(out)
            if k == 'obs':
                out = tf.cast(v, self.compute_dtype)
                out_accu.append(out)
        out = tf.math.accumulate_n(out_accu)
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


class ActionNet(tf.keras.layers.Layer):
    def __init__(self, env, categorical, entropy_contrib):
        super(ActionNet, self).__init__()
        self.entropy_contrib, self.categorical, self.is_discrete = tf.constant(entropy_contrib, self.compute_dtype), categorical, False

        if isinstance(env.action_space, gym.spaces.Discrete): num_components, event_shape, self.is_discrete = env.action_space.n, [1,], True
        elif isinstance(env.action_space, gym.spaces.Box): event_shape = list(env.action_space.shape); num_components = np.prod(event_shape).item()

        if categorical: params_size, self.dist = util.Categorical.params_size(num_components, event_shape), util.Categorical(num_components, event_shape)
        else: num_components *= 2; params_size, self.dist = util.MixtureLogistic.params_size(num_components, event_shape), util.MixtureLogistic(num_components, event_shape)

        self.net_blocks, self.net_LSTM, inp, mid, evo = 1, False, 256, 256, 32
        self.net_arch = "AN[inD{}-{:02d}{}D{}-cmp{}]".format(inp, self.net_blocks, ('LS+' if self.net_LSTM else ''), mid, num_components)

        self.layer_flatten = tf.keras.layers.Flatten()
        self.layer_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='dense_in')
        self.layer_lstm, self.layer_dense = [], []
        for i in range(self.net_blocks):
            if self.net_LSTM: self.layer_lstm.append(tf.keras.layers.LSTM(mid, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i)))
            self.layer_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='dense_{:02d}'.format(i)))
        
        if categorical: self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, name='dense_logits_out')
        else:
            self.split_cats, loc_scale_size_each = tf.constant(num_components, tf.int32), int((params_size-num_components)/2)
            self.layer_cont_cats, self.layer_cont_loc, self.layer_cont_scale = tf.keras.layers.Dense(num_components), tf.keras.layers.Dense(loc_scale_size_each), tf.keras.layers.Dense(loc_scale_size_each)

    def reset_states(self):
        if self.net_LSTM:
            for i in range(self.net_blocks): self.layer_lstm[i].reset_states()
    @tf.function
    def call(self, inputs, training=None):
        out_accu = []
        for k,v in inputs.items():
            if k == 'obs': out = tf.cast(v, self.compute_dtype); out_accu.append(out)
            if k == 'obs_pred': out = tf.cast(v, self.compute_dtype); out_accu.append(out)
        out = tf.concat(out_accu, 1)
        out = self.layer_flatten(out)
        out = self.layer_dense_in(out)
        for i in range(self.net_blocks):
            if self.net_LSTM: out = tf.squeeze(self.layer_lstm[i](tf.expand_dims(out, axis=0)), axis=0)
            out = self.layer_dense[i](out)
        if self.categorical: out = self.layer_dense_logits_out(out)
        else: out = util.combine_logits(out, self.layer_cont_cats, self.layer_cont_loc, self.layer_cont_scale, self.split_cats)
        
        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print('action net out:', out)
        return out

    def loss(self, dist, targets, advantages):
        targets = tf.cast(targets, dist.dtype)
        loss = -dist.log_prob(targets)
        loss = loss * advantages # * 1e-2
        if self.categorical:
            entropy = dist.entropy()
            loss = loss - entropy * self.entropy_contrib # "Soft Actor Critic" = try increase entropy

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
            if k == 'obs': out = tf.cast(v, self.compute_dtype); out_accu.append(out)
            if k == 'obs_pred': out = tf.cast(v, self.compute_dtype); out_accu.append(out)
        out = tf.concat(out_accu, 1)
        out = self.layer_flatten(out)
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
    def __init__(self, arch, env, max_episodes, max_steps, learn_rate, entropy_contrib, returns_disc, returns_std, action_cat, latent_size, latent_cat):
        super(GeneralAI, self).__init__()
        self.max_episodes, self.max_steps, self.returns_disc, self.returns_std = tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(returns_disc, tf.float64), returns_std
        self.float_maxroot = tf.constant(tf.math.sqrt(tf.dtypes.as_dtype(self.compute_dtype).max), self.compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(self.compute_dtype).eps, self.compute_dtype)

        self.env = env
        self.obs_dtype = tf.dtypes.as_dtype(env.observation_space.dtype)
        self.action_dtype = tf.dtypes.as_dtype(env.action_space.dtype)
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_shape = [1,]
            self.action_min = tf.constant(0, self.compute_dtype)
            self.action_max = tf.constant(env.action_space.n-1, self.compute_dtype)
        elif isinstance(env.action_space, gym.spaces.Box):
            action_shape = list(env.action_space.shape)
            self.action_min = tf.constant(env.action_space.low[...,0], self.compute_dtype)
            self.action_max = tf.constant(env.action_space.high[...,0], self.compute_dtype)

        inputs = {'obs':tf.zeros([1]+list(env.observation_space.shape),self.obs_dtype), 'reward':tf.constant([[0]],tf.float64), 'done':tf.constant([[False]],tf.bool)}
        if arch=='DREAM':
            latent_size = env.observation_space.shape[0] # TODO hack until using RepNet
            self.latent_dtype = tf.int32 if latent_cat else self.compute_dtype
            self.latent_zero, self.latent_invar = tf.constant(tf.zeros((1,latent_size), self.latent_dtype)), tf.TensorShape([None,latent_size])
            inputs_trans = {'action':tf.zeros([1]+action_shape,self.action_dtype), 'obs':self.latent_zero}
            self.trans = TransNet(latent_size, latent_cat); outputs = self.trans(inputs_trans)
            inputs['obs_pred'] = self.latent_zero
        self.action = ActionNet(env, action_cat, entropy_contrib); outputs = self.action(inputs)
        self.value = ValueNet(env); outputs = self.value(inputs)

        self.action_storage_dtype = tf.int32 if action_cat else self.compute_dtype
        self.reward_episode = tf.Variable(0, dtype=tf.float64, trainable=False)
        self.discounted_sum = tf.Variable(0, dtype=tf.float64, trainable=False)

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=self.float_eps)


    def env_reset(self):
        obs, reward, done = self.env.reset(), 0.0, False
        return np.expand_dims(obs,0), np.expand_dims(np.asarray(reward, np.float64),(0,1)), np.expand_dims(np.asarray(done, np.bool),(0,1))
    def env_step(self, action):
        obs, reward, done, _ = self.env.step(action)
        return np.expand_dims(obs,0), np.expand_dims(np.asarray(reward, np.float64),(0,1)), np.expand_dims(np.asarray(done, np.bool),(0,1))

    def action_discretize(self, action):
        action = tf.math.round(action)
        action = tf.clip_by_value(action, self.action_min, self.action_max)
        action = tf.cast(action, self.action_dtype)
        return action

    def calc_returns(self, rewards):
        self.discounted_sum.assign(0)
        n = tf.shape(rewards)[0]
        rewards = rewards[::-1]
        returns = tf.TensorArray(tf.float64, size=n)
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = self.discounted_sum.value() * self.returns_disc + reward
            returns = returns.write(i, discounted_sum)
            self.discounted_sum.assign(discounted_sum)
        returns = returns.stack()[::-1]

        if self.returns_std: returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.float_eps))
        returns = tf.cast(returns, self.compute_dtype)
        return tf.stop_gradient(returns)


    @tf.function
    def AC_actor(self):
        print("tracing -> GeneralAI AC_actor")
        inputs, outputs = {}, {}

        obs = tf.TensorArray(self.obs_dtype, size=0, dynamic_size=True)
        actions = tf.TensorArray(self.action_storage_dtype, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        
        inputs['obs'], inputs['reward'], inputs['done'] = tf.numpy_function(self.env_reset, [], (self.obs_dtype, tf.float64, tf.bool))

        self.action.reset_states()
        for step in tf.range(self.max_steps):
            obs = obs.write(step, inputs['obs'][-1])

            action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)
            action = action_dist.sample()
            actions = actions.write(step, action[-1])

            if not self.action.categorical and self.action.is_discrete: action = self.action_discretize(action)
            action = tf.squeeze(action)
            inputs['obs'], inputs['reward'], inputs['done'] = tf.numpy_function(self.env_step, [action], (self.obs_dtype, tf.float64, tf.bool))

            rewards = rewards.write(step, inputs['reward'][-1][0])
            if inputs['done'][-1][0]: break
        
        outputs['obs'], outputs['actions'], outputs['rewards'] = obs.stack(), actions.stack(), rewards.stack()
        return outputs
    @tf.function
    def AC_learner(self, inputs, training=True):
        print("tracing -> GeneralAI AC_learner")

        self.action.reset_states()
        action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)

        self.value.reset_states()
        values = self.value(inputs)
        values = tf.squeeze(values, axis=-1)

        returns = self.calc_returns(inputs['rewards'])
        advantages = returns - values
        
        loss = {}
        # loss['action'] = self.action.loss(action_dist, inputs['actions'], advantages)
        loss['action'] = self.action.loss(action_dist, inputs['actions'], inputs['rewards'])
        loss['value'] = self.value.loss(advantages)
        loss['total'] = loss['action'] + loss['value']

        loss['returns'] = returns
        loss['advantages'] = advantages
        return loss
    @tf.function
    def AC_run(self):
        print("tracing -> GeneralAI AC_run")
        metrics = {'rewards_total':tf.float64,'steps':tf.int32,'loss_total':self.compute_dtype,'loss_action':self.compute_dtype,'loss_value':self.compute_dtype,'returns':self.compute_dtype,'advantages':self.compute_dtype}
        for k in metrics.keys(): metrics[k] = tf.TensorArray(metrics[k], size=0, dynamic_size=True)

        for episode in tf.range(self.max_episodes):
            outputs = self.AC_actor()
            with tf.GradientTape() as tape:
                loss = self.AC_learner(outputs)
            gradients = tape.gradient(loss['total'], self.action.trainable_variables + self.value.trainable_variables)
            # isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(gradients[0]), tf.math.is_inf(gradients[0])))
            # if isinfnan > 0: tf.print('\ngradients', gradients[0]); break
            self._optimizer.apply_gradients(zip(gradients, self.action.trainable_variables + self.value.trainable_variables))
            

            metrics['rewards_total'] = metrics['rewards_total'].write(episode,  tf.math.reduce_sum(outputs['rewards']))
            metrics['steps'] = metrics['steps'].write(episode, tf.shape(outputs['rewards'])[0])
            metrics['loss_total'] = metrics['loss_total'].write(episode, tf.math.reduce_mean(loss['total']))
            metrics['loss_action'] = metrics['loss_action'].write(episode, tf.math.reduce_mean(loss['action']))
            metrics['loss_value'] = metrics['loss_value'].write(episode, tf.math.reduce_mean(loss['value']))
            metrics['returns'] = metrics['returns'].write(episode, tf.math.reduce_mean(loss['returns']))
            metrics['advantages'] = metrics['advantages'].write(episode, tf.math.reduce_mean(loss['advantages']))
            # for k in metrics.keys(): tf.print(k, metrics[k].read(episode), end=' ')
        for k in metrics.keys(): metrics[k] = metrics[k].stack()
        return metrics
        


    @tf.function
    def DREAM_actor(self):
        print("tracing -> GeneralAI DREAM_actor")
        inputs, outputs = {}, {}

        obs = tf.TensorArray(self.obs_dtype, size=0, dynamic_size=True)
        actions = tf.TensorArray(self.action_storage_dtype, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        obs_pred = tf.TensorArray(self.latent_dtype, size=0, dynamic_size=True)
        
        inputs['obs'], inputs['reward'], inputs['done'] = tf.numpy_function(self.env_reset, [], (self.obs_dtype, tf.float64, tf.bool))
        inputs['obs_pred'] = self.latent_zero

        self.trans.reset_states(); self.action.reset_states()
        for step in tf.range(self.max_steps):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs['obs_pred'], self.latent_invar)])
            obs = obs.write(step, inputs['obs'][-1])

            inputs['obs'] = tf.cast(inputs['obs'], self.latent_dtype) # RepNet

            trans_logits = self.trans(inputs); trans_dist = self.trans.dist(trans_logits)
            inputs['obs_pred'] = trans_dist.sample()
            obs_pred = obs_pred.write(step, inputs['obs_pred'][-1])
            
            action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)
            action = action_dist.sample()
            actions = actions.write(step, action[-1])

            if not self.action.categorical and self.action.is_discrete: action = self.action_discretize(action)
            action = tf.squeeze(action)
            inputs['obs'], inputs['reward'], inputs['done'] = tf.numpy_function(self.env_step, [action], (self.obs_dtype, tf.float64, tf.bool))

            rewards = rewards.write(step, inputs['reward'][-1][0])
            if inputs['done'][-1][0]: break
        
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['obs_pred'] = obs.stack(), actions.stack(), rewards.stack(), obs_pred.stack()
        return outputs

    @tf.function
    def DREAM_learner(self, inputs_, training=True):
        print("tracing -> GeneralAI DREAM_learner")
        inputs = inputs_.copy()

        inputs['obs'] = tf.cast(inputs['obs'], self.latent_dtype) # RepNet

        # self.trans.reset_states()
        # trans_logits = self.trans(inputs); trans_dist = self.trans.dist(trans_logits)
        # # inputs_pred['obs'] = trans_dist.sample()

        self.action.reset_states()
        action_logits = self.action(inputs); action_dist = self.action.dist(action_logits)

        self.value.reset_states()
        values = self.value(inputs)
        values = tf.squeeze(values, axis=-1)

        returns = self.calc_returns(inputs['rewards'])
        advantages = returns - values
        
        loss = {}
        # future = tf.roll(inputs['obs'], -1, axis=0) # GenNet
        # loss['trans'] = self.trans.loss(trans_dist, future)
        # loss['action'] = self.action.loss(action_dist, inputs['actions'], advantages)
        loss['action'] = self.action.loss(action_dist, inputs['actions'], inputs['rewards'])
        loss['value'] = self.value.loss(advantages)
        loss['total'] = loss['action'] + loss['value']
        # loss['total'] = loss['trans'] + loss['action'] + loss['value']

        loss['returns'] = returns
        loss['advantages'] = advantages
        return loss

    @tf.function
    def DREAM_run(self):
        print("tracing -> GeneralAI DREAM_run")
        metrics = {'rewards_total':tf.float64,'steps':tf.int32,'loss_total':self.compute_dtype,'loss_action':self.compute_dtype,'loss_value':self.compute_dtype,'returns':self.compute_dtype,'advantages':self.compute_dtype,'trans':self.compute_dtype}
        for k in metrics.keys(): metrics[k] = tf.TensorArray(metrics[k], size=0, dynamic_size=True)

        for episode in tf.range(self.max_episodes):
            outputs = self.DREAM_actor()
            with tf.GradientTape() as tape:
                loss = self.DREAM_learner(outputs)
            gradients = tape.gradient(loss['total'], self.action.trainable_variables + self.value.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.action.trainable_variables + self.value.trainable_variables))
            # gradients = tape.gradient(loss['total'], self.trans.trainable_variables + self.action.trainable_variables + self.value.trainable_variables)
            # self._optimizer.apply_gradients(zip(gradients, self.trans.trainable_variables + self.action.trainable_variables + self.value.trainable_variables))
            
            with tf.GradientTape() as tape:
                outputs['obs'] = tf.cast(outputs['obs'], self.compute_dtype) # RepNet
                self.trans.reset_states()
                trans_logits = self.trans(outputs); trans_dist = self.trans.dist(trans_logits)
                future = tf.roll(outputs['obs'], -1, axis=0) # GenNet
                loss['trans'] = self.trans.loss(trans_dist, future)
            gradients = tape.gradient(loss['trans'], self.trans.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trans.trainable_variables))

            metrics['rewards_total'] = metrics['rewards_total'].write(episode,  tf.math.reduce_sum(outputs['rewards']))
            metrics['steps'] = metrics['steps'].write(episode, tf.shape(outputs['rewards'])[0])
            metrics['loss_total'] = metrics['loss_total'].write(episode, tf.math.reduce_mean(loss['total']))
            metrics['loss_action'] = metrics['loss_action'].write(episode, tf.math.reduce_mean(loss['action']))
            metrics['loss_value'] = metrics['loss_value'].write(episode, tf.math.reduce_mean(loss['value']))
            metrics['returns'] = metrics['returns'].write(episode, tf.math.reduce_mean(loss['returns']))
            metrics['advantages'] = metrics['advantages'].write(episode, tf.math.reduce_mean(loss['advantages']))
            metrics['trans'] = metrics['trans'].write(episode, tf.math.reduce_mean(loss['trans']))
            # for k in metrics.keys(): tf.print(k, metrics[k].read(episode), end=' ')
        for k in metrics.keys(): metrics[k] = metrics[k].stack()
        return metrics



def params(): pass
max_episodes = 100
learn_rate = 1e-5
entropy_contrib = 1e-8
returns_disc = 0.99
returns_std = False
action_cat = True
latent_size = 1
latent_cat = True
trader, trader_env, trader_speed = False, 3, 180.0

machine, device = 'dev', 0

# env_name, max_steps, env = 'CartPole', 201, gym.make('CartPole-v0'); env.observation_space.dtype = np.dtype('float64')
# env_name, max_steps, env = 'LunarLand', 1001, gym.make('LunarLander-v2')
# env_name, max_steps, env = 'LunarLandCont', 1001, gym.make('LunarLanderContinuous-v2')
# import envs_local.random as env_; env_name, max_steps, env = 'TestRnd', 128, env_.RandomEnv()
import envs_local.data as env_; env_name, max_steps, env = 'DataShkspr', 128, env_.DataEnv('shkspr')
# import envs_local.data as env_; env_name, max_steps, env = 'DataMnist', 128, env_.DataEnv('mnist')
# import envs_local.bipedal_walker as env_; env_name, max_steps, env = 'BipedalWalker', 100, env_.BipedalWalker()
# import gym_trader; env_name, max_steps, trader, env = 'Trader2', 100, True, gym.make('Trader-v0', agent_id=device, env=trader_env, speed=trader_speed)

# import envs_local.async_wrapper as env_async_wrapper; env_name, env = env_name+'-Asyn', env_async_wrapper.AsyncWrapperEnv(env)

arch = 'AC'
# arch = 'DREAM'

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
        model = GeneralAI(arch, env, max_episodes=max_episodes, max_steps=max_steps, learn_rate=learn_rate, entropy_contrib=entropy_contrib, returns_disc=returns_disc, returns_std=returns_std, action_cat=action_cat, latent_size=latent_size, latent_cat=latent_cat)
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
        if arch=='AC': metrics = model.AC_run()
        if arch=='DREAM': metrics = model.DREAM_run()
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds

        # # TODO save models
        # model.save_weights(model_file)
        
        ## metrics
        name, name_arch = "{}-{}-a{}-{}".format(name, machine, device, time.strftime("%Y_%m_%d-%H-%M")), ""
        for net in model.layers: name_arch += "   "+net.net_arch
        total_steps = np.sum(metrics['steps'])
        step_time = total_time/total_steps
        title = "{} {}\ntime:{}    steps:{}    t/s:{:.8f}     |     lr:{}    en:{}    std:{}".format(name, name_arch, util.print_time(total_time), total_steps, step_time, learn_rate, entropy_contrib, returns_std); print(title)

        import matplotlib as mpl
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'lightblue', 'green', 'lime', 'red', 'lavender', 'turquoise', 'cyan', 'magenta', 'salmon', 'yellow', 'gold', 'black', 'brown', 'purple', 'pink', 'orange', 'teal', 'coral', 'darkgreen', 'tan'])
        plt.figure(num=name, figsize=(34, 16), tight_layout=True)
        xrng, i, vplts = np.arange(0, max_episodes, 1), 0, 7
        if arch=='DREAM': vplts = 8

        rows = 2; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
        metric_name = 'rewards_total'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left'); plt.title(title)

        rows = 4; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
        metric_name = 'loss_total'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        metric_name = 'loss_action'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        metric_name = 'loss_value'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        metric_name = 'returns'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        metric_name = 'advantages'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left')
        
        if arch=='DREAM':
            rows = 1; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
            metric_name = 'trans'; metric = np.asarray(metrics[metric_name], np.float64)
            plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
            plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left')
        
        rows = 1; plt.subplot2grid((vplts, 1), (i, 0), rowspan=rows); i+=rows; plt.grid(axis='y',alpha=0.3)
        metric_name = 'steps'; metric = np.asarray(metrics[metric_name], np.float64)
        plt.plot(xrng, talib.EMA(metric, timeperiod=max_episodes//10+2), alpha=1.0, label=metric_name); plt.plot(xrng, metric, alpha=0.3)
        plt.ylabel('value'); plt.xlabel('episode'); plt.legend(loc='upper left')

        plt.show()
