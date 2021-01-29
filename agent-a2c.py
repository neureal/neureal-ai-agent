import time, os
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
import talib
import gym, gym_trader
import util

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


class ActorCriticAI(tf.keras.Model):
    def __init__(self, env, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        super().__init__('mlp_policy')
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma, self.value_c, self.entropy_c = tf.constant(gamma, dtype=self.compute_dtype), tf.constant(value_c, dtype=self.compute_dtype), tf.constant(entropy_c, dtype=self.compute_dtype)
        # self.float_max = tf.constant(tf.dtypes.as_dtype(self.compute_dtype).max, dtype=self.compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(tf.dtypes.as_dtype(self.compute_dtype).max), dtype=self.compute_dtype)

        self._optimizer = tf.keras.optimizers.Adam(lr=lr)
        self._optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self._optimizer)


        self.action_num_components = 16
        self.action_params_size, self.action_dist, self.action_is_discrete = 0, {}, True
        # self.action_params_size, self.action_dist, self.action_is_discrete = util.gym_action_dist(env.action_space, dtype=self.compute_dtype, num_components=self.action_num_components)
        cat_dtype = tf.int32
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_params_size = env.action_space.n
            if env.action_space.n < 2: self.action_dist = tfp.layers.IndependentBernoulli(1, sample_dtype=tf.bool)
            else: self.action_dist = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Categorical(logits=input, dtype=cat_dtype))
        elif isinstance(env.action_space, gym.spaces.Box):
            event_shape = list(env.action_space.shape)
            event_size = np.prod(event_shape).item()
            if env.action_space.dtype == np.uint8:
                self.action_params_size = event_size * 256
                if event_size == 1: self.action_dist = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Categorical(logits=input, dtype=cat_dtype))
                else:
                    params_shape = event_shape+[256]
                    self.action_dist = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Independent(
                        tfp.distributions.Categorical(logits=tf.reshape(input, tf.concat([tf.shape(input)[:-1], params_shape], axis=0)), dtype=cat_dtype), reinterpreted_batch_ndims=len(event_shape)
                    ))
            elif env.action_space.dtype == np.float32 or env.action_space.dtype == np.float64:
                if len(event_shape) == 1: # arbitrary, but with big event shapes the paramater size is HUGE
                    # self.action_params_size = tfp.layers.MixtureSameFamily.params_size(self.action_num_components, component_params_size=tfp.layers.MultivariateNormalTriL.params_size(event_size))
                    # self.action_dist = tfp.layers.MixtureSameFamily(self.action_num_components, tfp.layers.MultivariateNormalTriL(event_size))
                    self.action_params_size = tfp.layers.MixtureLogistic.params_size(self.action_num_components, event_shape)
                    self.action_dist = tfp.layers.MixtureLogistic(self.action_num_components, event_shape)
                else:
                    self.action_params_size = tfp.layers.MixtureNormal.params_size(self.action_num_components, event_shape)
                    self.action_dist = tfp.layers.MixtureNormal(self.action_num_components, event_shape)
                # self.bijector = tfp.bijectors.Sigmoid(low=-1.0, high=1.0)
                self.action_is_discrete = False
        # TODO expand to handle Tuple+Dict, make so discrete and continuous can be output at the same time
        # event_shape/event_size = action_size['action_dist_pair'] + action_size['action_dist_percent']
        elif isinstance(env.action_space, gym.spaces.Tuple):
            for space in env.action_space.spaces:
                self.action_params_size += space.n
                print(space)
            quit(0)

        # TODO expand the Dict of observation types (env.observation_space) to auto make input networks
        self.obs_is_vec = (len(env.observation_space.shape) == 1)
        # self.obs_is_vec = util.gym_obs_embed(env.observation_space)

        self.net_DNN, self.net_LSTM, inp, mid, evo = 0, 1, 1024, 512, 16
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 4, 1, 2048, 1024, 32
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 8, 1, 4096, 2048, 32
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 1, 1, 128, 64, 16
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 2, 2, 256, 128, 16
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 4, 4, 1024, 512, 32
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 4, 4, 2048, 1024, 32
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 6, 6, 2048, 1024, 32
        self.net_arch = "{}inD{}-{:02d}D{}-{:02d}LS{}-out{}".format('' if self.obs_is_vec else 'Conv2D-', inp, self.net_DNN, mid, self.net_LSTM, mid, 'D' if self.action_is_discrete else 'Cont'+str(self.action_num_components))

        self.layer_action_dense, self.layer_action_lstm, self.layer_value_dense, self.layer_value_lstm = [], [], [], []
        self.layer_flatten = tf.keras.layers.Flatten()
        # if not self.obs_is_vec: self.layer_conv2d_in = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=util.EvoNormS0(32), name='conv2d_in')

        ## action network
        # if not self.obs_is_vec: self.layer_action_conv2d_in = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=util.EvoNormS0(evo), name='action_conv2d_in')
        self.layer_action_dense_in = tf.keras.layers.Dense(inp, activation=util.EvoNormS0(evo), use_bias=False, name='action_dense_in')
        for i in range(self.net_DNN): self.layer_action_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='action_dense_{:02d}'.format(i)))
        for i in range(self.net_LSTM): self.layer_action_lstm.append(tf.keras.layers.LSTM(mid, return_sequences=True, stateful=True, name='action_lstm_{:02d}'.format(i)))
        if not self.action_is_discrete: self.layer_action_dense_cont = tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='action_dense_cont')
        # self.layer_action_deconv1d_logits_out = tf.keras.layers.Conv1DTranspose(self.action_params_size/4, 4, name='action_deconv1d_logits_out')
        self.layer_action_dense_logits_out = tf.keras.layers.Dense(self.action_params_size, name='action_dense_logits_out')

        ## value network
        # if not self.obs_is_vec: self.layer_value_conv2d_in = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=util.EvoNormS0(evo), name='value_conv2d_in')
        self.layer_value_dense_in = tf.keras.layers.Dense(inp, use_bias=False, name='value_dense_in')
        for i in range(self.net_DNN+2): self.layer_value_dense.append(tf.keras.layers.Dense(mid, activation=util.EvoNormS0(evo), use_bias=False, name='value_dense_{:02d}'.format(i)))
        for i in range(self.net_LSTM): self.layer_value_lstm.append(tf.keras.layers.LSTM(mid, return_sequences=True, stateful=True, name='value_lstm_{:02d}'.format(i)))
        self.layer_value_dense_out = tf.keras.layers.Dense(1, name='value_dense_out')

        # pre build model
        sample = tf.convert_to_tensor([env.observation_space.sample()], dtype=self.compute_dtype)
        # action, value = self(sample)
        outputs = self(sample)
        self.reset_states() # needed because the previous call unnormally advances the state!

    @tf.function
    def call(self, inputs, training=None):
        print("tracing -> call")
        if not self.obs_is_vec: inputs = self.layer_flatten(inputs)
        # if not self.obs_is_vec:
        #     inputs = self.layer_conv2d_in(inputs)
        #     inputs = self.layer_flatten(inputs)

        ## action network
        action = self.layer_action_dense_in(inputs)
        # if self.obs_is_vec:
        #     action = self.layer_action_dense_in(inputs)
        # else:
        #     action = self.layer_action_conv2d_in(inputs)
        #     action = self.layer_flatten(action)
        #     action = self.layer_action_dense_in(action)
        for i in range(self.net_DNN): action = self.layer_action_dense[i](action)
        for i in range(self.net_LSTM): action = tf.squeeze(self.layer_action_lstm[i](tf.expand_dims(action, axis=0)), axis=0)
        if not self.action_is_discrete: action = self.layer_action_dense_cont(action)
        action = self.layer_action_dense_logits_out(action)
        # batch_size = inputs.shape[0]
        # action = tf.expand_dims(action, axis=1)
        # action = self.layer_deconv1d_logits_out(action)
        # action = tf.reshape(action, (batch_size, self.action_params_size))

        ## value network
        value = self.layer_value_dense_in(inputs)
        # if self.obs_is_vec:
        #     value = self.layer_value_dense_in(inputs)
        # else:
        #     value = self.layer_value_conv2d_in(inputs)
        #     value = self.layer_flatten(value)
        #     value = self.layer_value_dense_in(value)
        for i in range(self.net_DNN+2): value = self.layer_value_dense[i](value)
        for i in range(self.net_LSTM): value = tf.squeeze(self.layer_value_lstm[i](tf.expand_dims(value, axis=0)), axis=0)
        value = self.layer_value_dense_out(value)

        return action, value

    # TODO convert to tf graph? or share this processing with CPU
    # TODO use numba to make this faster on CPU
    # @tf.function
    def calc_returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        # test = np.asarray(next_value)
        returns = np.append(np.zeros_like(rewards), [next_value], axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + returns[t + 1] * self.gamma * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        # advantages = np.abs(advantages) # for MAE, could be np.square() for MSE
        return returns, advantages

    def _loss_action(self, actions, action_logits, advantages): # targets, output (layer_action_dense_logits_out, returns - values)
        dist = self.action_dist(action_logits)
        # dist = tfp.distributions.TransformedDistribution(distribution=dist, bijector=self.bijector)
        actions = tf.cast(actions, dtype=dist.dtype)
        if dist.event_shape.rank == 0: actions = tf.squeeze(actions, axis=-1)
        if self.action_is_discrete:
            loss = -dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            loss = -dist.log_prob(actions)
            entropy = dist.sample(self.action_num_components)
            entropy = -dist.log_prob(entropy)
            # entropy = util.replace_infnan(entropy, self.float_maxroot)
            entropy = tf.reduce_mean(entropy, axis=0)

        loss = tf.math.multiply(loss, advantages) # sample_weight
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        loss = loss - entropy * self.entropy_c
        loss = util.replace_infnan(loss, self.float_maxroot)
        return loss # shape = (batch size,)

    # loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # def _loss_value(self, returns, values): # targets, output (returns, layer_value_dense_out)
    # def _loss_value(self, returns, values, advantages): # targets, output (returns, layer_value_dense_out)
    def _loss_value(self, advantages):
        # Value loss is typically MSE between value estimates and returns.
        # returns = tf.expand_dims(returns, 1)
        # loss_value = self.loss_fn(returns, values)
        # loss_value = tf.keras.losses.mean_squared_error(returns, values)
        # loss_value = tf.math.square(advantages)
        # loss_value = loss_value * self.value_c
        loss_value = tf.math.square(advantages) * self.value_c
        return loss_value # shape = (batch size,)

    @tf.function
    # def train(self, inputs, actions, returns, advantages, dones):
    # def train(self, inputs, actions, returns, advantages):
    def _train(self, inputs, actions, returns):
        print("tracing -> _train")
        self.reset_states()
        with tf.GradientTape() as tape:
            action_logits, values = self(inputs, training=True) # redundant, but needed to be able to calculate the gradients backwards
            values = tf.squeeze(values, axis=-1)
            advantages = returns - values # should calculate returns inside gradient tape so can do backprop on it!!
            # advantages = tf.math.square(advantages)
            # advantages = tf.math.abs(advantages) # CRASHES!!
            # isneg = tf.math.equal(tf.math.sign(advantages),-1.0)
            # advantages = tf.where(isneg, advantages*-1.0, advantages)

            loss_action = self._loss_action(actions, action_logits, advantages)
            # loss_value = self._loss_value(returns, values) # redundant, but needed to be able to calculate the gradients backwards
            # loss_value = self._loss_value(returns, new_values, advantages)
            loss_value = self._loss_value(advantages)
            loss_total = loss_action + loss_value
        gradients = tape.gradient(loss_total, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.reset_states()

        # # ??? loop through timesteps instead of batch so can keep batch size = 1 with LSTM and take out return_sequences=True, add self.reset_states() on episode end
        # # TODO save state and reset state here, then restore state before leaving
        # batch_size = inputs.shape[0]
        # for i in tf.range(batch_size):
        #     with tf.GradientTape() as tape:
        #         action_logits, value = self(inputs[None, i], training=True)
        #         loss_action = self._loss_action(actions[None, i], advantages[None, i], action_logits)
        #         loss_value = self._loss_value(returns[None, i], value)
        #         loss_total = loss_action + loss_value
        #     gradients = tape.gradient(loss_total, self.trainable_variables)
        #     self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        #     if dones[i]: self.reset_states()

        return tf.reduce_mean(loss_total), tf.reduce_mean(loss_action), tf.reduce_mean(loss_value)

    def train(self, obs, actions, returns):
        obs = tf.convert_to_tensor(obs, dtype=self.compute_dtype)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns, dtype=self.compute_dtype)
        loss_total_cur, loss_action_cur, loss_value_cur = self._train(obs, actions, returns)
        return loss_total_cur.numpy(), loss_action_cur.numpy(), loss_value_cur.numpy()


    @tf.function
    def _action_value(self, inputs):
        action_logits, value = self(inputs)
        dist = self.action_dist(action_logits)
        # dist = tfp.distributions.TransformedDistribution(distribution=dist, bijector=self.bijector) # doesn't sample with float64
        action = dist.sample()

        # action = {}
        # for name, dist in self.action_dist:
        #     dist = dist(action_logits[])
        #     action[name] = tf.squeeze(dist.sample())

        return action, value

    def action_value(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=self.compute_dtype) # add single batch
        action, value = self._action_value(obs)
        action, value = action.numpy().squeeze(), value.numpy().squeeze() # get rid of single batch
        # print("action {} value {}".format(action, value))
        return action, value


class AgentA2C:
    def __init__(self, model):
        self.model = model

    def train(self, env, render=False, batch_sz=64, updates=250):
        if isinstance(env.action_space, gym.spaces.Discrete): action_shape = [1]
        elif isinstance(env.action_space, gym.spaces.Box): action_shape = list(env.action_space.shape)
        # Storage helpers for a single batch of data.
        observations = np.empty([batch_sz]+list(env.observation_space.shape), dtype=env.observation_space.dtype)
        actions = np.empty([batch_sz]+action_shape, dtype=env.action_space.dtype)
        values = np.empty((batch_sz,), dtype=model.compute_dtype)
        rewards = np.empty((batch_sz,), dtype=model.compute_dtype)
        dones = np.empty((batch_sz,), dtype=np.bool)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        next_obs = env.reset()
        if render: env.render()

        loss_total, loss_action, loss_value, loss_total_cur, loss_action_cur, loss_value_cur = [], [], [], 0.0, 0.0, 0.0
        epi_total_rewards, epi_steps, epi_avg_reward, epi_end_reward, epi_avg_rewards, epi_end_rewards, epi_sim_times = [0.0], 0, 0, 0, [], [], []
        update, finished, early_quit = 0, False, False
        t_real_start, t_sim_total, t_sim_epi_start  = time.time(), 0.0, np.mean(next_obs[0])
        steps_total, t_steps_total = 0, 0.0

        while update < updates or not finished:
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs)
                
                step_time_start = time.perf_counter_ns()
                next_obs, rewards[step], dones[step], _ = env.step(np.squeeze(actions[step]))
                t_steps_total += (time.perf_counter_ns() - step_time_start) / 1e9 # seconds
                steps_total += 1
                if render:
                    if not env.render(): epi_total_rewards = epi_total_rewards[:-1]; early_quit = True; break

                epi_total_rewards[-1] += rewards[step]
                epi_steps += 1
                if dones[step]:
                    epi_end_reward = np.expm1(rewards[step])
                    epi_end_rewards.append(epi_end_reward)
                    epi_avg_reward = np.expm1(epi_total_rewards[-1] / epi_steps)
                    epi_avg_rewards.append(epi_avg_reward)
                    t_sim_epi = np.mean(next_obs[0]) - t_sim_epi_start
                    epi_sim_times.append(t_sim_epi / 60)
                    loss_total.append(loss_total_cur); loss_action.append(loss_action_cur); loss_value.append(loss_value_cur)
                    print("DONE episode #{:03d}  {} sim-epi-time {:10.2f} total-reward {:10.2f} avg-reward {:10.2f} end-reward".format(len(epi_total_rewards)-1, util.print_time(t_sim_epi), epi_total_rewards[-1], epi_avg_reward, epi_end_reward))
                    # TODO train/update after every done
                    if update >= updates-1: finished = True; break
                    epi_total_rewards.append(0.0)
                    epi_steps = 0
                    t_sim_total += t_sim_epi

                    next_obs = env.reset()
                    if render: env.render()
                    t_sim_epi_start = np.mean(next_obs[0])
            if early_quit: break

            self.model.reset_states() # needed because otherwise this next call unnormally advances the state, ineffecient!
            _, next_value = self.model.action_value(next_obs)
            # returns ~ [7,6,5,4,3,2,1,0] for reward = 1 per step, advantages = returns - values output
            returns, advantages = self.model.calc_returns_advantages(rewards, dones, values, next_value)

            # loss_total_cur, loss_action_cur, loss_value_cur = self.model.train(observations, actions, returns, advantages, dones)
            # loss_total_cur, loss_action_cur, loss_value_cur = self.model.train(observations, actions, returns, advantages)
            loss_total_cur, loss_action_cur, loss_value_cur = self.model.train(observations, actions, returns)

            print("---> update [{:03d}/{:03d}]                                                                                            {:16.8f} loss_total {:16.8f} loss_action {:16.8f} loss_value".format(update, updates, loss_total_cur, loss_action_cur, loss_value_cur))
            # print("---> update [{:03d}/{:03d}]  {:10.2f} total-reward {:10.2f} avg-reward {:10.2f} last-reward  {:16.8f} loss_total {:16.8f} loss_action {:16.8f} loss_value".format(update, updates, epi_total_rewards[-1], epi_avg_reward, np.expm1(rewards[step]), loss_total_cur, loss_action_cur, loss_value_cur))
            update += 1

        epi_num = len(epi_end_rewards)
        t_sim_total += np.mean(next_obs[0]) - t_sim_epi_start
        t_avg_sim_epi = (t_sim_total / epi_num) if epi_num > 0 else 0

        t_real_total = time.time() - t_real_start
        t_avg_step = (t_steps_total / steps_total) if steps_total > 0 else 0

        return epi_num, np.asarray(epi_total_rewards), np.asarray(epi_end_rewards), np.asarray(epi_avg_rewards), np.asarray(epi_sim_times), t_sim_total, t_avg_sim_epi, t_real_total, t_avg_step, np.asarray(loss_total), np.asarray(loss_action), np.asarray(loss_value)

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        if render: env.render()
        while not done:
            action, _ = self.model.action_value(obs)
            obs, reward, done, _ = env.step(action)
            if render: env.render()
            ep_reward += reward
        return ep_reward



class Args(): pass
args = Args()
args.batch_size = 1024 # about 1.5 hrs @ 1000.0 speed
args.num_updates = 10 # roughly batch_size * num_updates = total steps, unless last episode is long
args.learning_rate = 1e-4 # start with 4 for rough train, 5 for fine tune and 6 for when trained
args.value_c = 0.9 # scaler for value loss contribution
args.entropy_c = 1e-4 # scaler for entropy loss contribution
args.render = False
args.plot_results = True

machine, device = 'dev', 0

import envs_local.bipedal_walker as env_bipedal_walker
import envs_local.car_racing as env_car_racing

if __name__ == '__main__':
    # env, model_name = gym.make('CartPole-v0'), "gym-A2C-CartPole"                                   # Box((4),-inf:inf,float32)         Discrete(2,int64)             200    100  195.0 # obs return float64 even though specs say float32?!?
    # env, model_name = gym.make('LunarLander-v2'), "gym-A2C-LunarLander"                             # Box((8),-inf:inf,float32)         Discrete(4,int64)             1000   100  200
    env, model_name = gym.make('LunarLanderContinuous-v2'), "gym-A2C-LunarLanderCont"               # Box((8),-inf:inf,float32)         Box((2),-1.0:1.0,float32)     1000   100  200
    # env, model_name = gym.make('BipedalWalker-v3'), "gym-A2C-BipedalWalker"                         # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name = env_bipedal_walker.BipedalWalker(), "gym-A2C-BipedalWalker"                   # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name = gym.make('BipedalWalkerHardcore-v3'), "gym-A2C-BipedalWalkerHardcore"         # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name = env_bipedal_walker.BipedalWalkerHardcore(), "gym-A2C-BipedalWalkerHardcore"   # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name = gym.make('CarRacing-v0'), "gym-A2C-CarRacing"                                 # Box((96,96,3),0:255,uint8)        Box((3),-1.0:1.0,float32)     1000   100  900 # MEMORY LEAK!
    # env, model_name = env_car_racing.CarRacing(), "gym-A2C-CarRacing"                               # Box((96,96,3),0:255,uint8)        Box((3),-1.0:1.0,float32)     1000   100  900 # MEMORY LEAK!
    # env, model_name = gym.make('QbertNoFrameskip-v4'), "gym-A2C-QbertNoFrameskip-v4"                # Box((210,160,3),0:255,uint8)      Discrete(6)                   400000 100  None
    # env, model_name = gym.make('BoxingNoFrameskip-v4'), "gym-A2C-BoxingNoFrameskip-v4"              # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name = gym.make('CentipedeNoFrameskip-v4'), "gym-A2C-CentipedeNoFrameskip-v4"        # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name = gym.make('PitfallNoFrameskip-v4'), "gym-A2C-PitfallNoFrameskip-v4"            # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name = gym.make('MontezumaRevengeNoFrameskip-v4'), "gym-A2C-MontezumaRevenge-v4"     # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name = gym.make('ReversedAddition3-v0'), "gym-A2C-ReversedAddition3"                 # Discrete(4)                       Tuple(Dis(4),Dis(2),Dis(3))
    # env, model_name = gym.make('Trader-v0', agent_id=device, env=2, speed=200.0), "gym-A2C-Trader2"

    # env.seed(0)
    with tf.device('/device:GPU:'+str(device)):
        model = ActorCriticAI(env, lr=args.learning_rate, value_c=args.value_c, entropy_c=args.entropy_c)
        model_name += "-{}-{}-a{}".format(model.net_arch, machine, device)

        model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
        if tf.io.gfile.exists(model_file):
            model.load_weights(model_file, by_name=True, skip_mismatch=True)
            print("LOADED model weights from {}".format(model_file)); loaded_model = True
        # print(model.call.pretty_printed_concrete_signatures()); quit(0)
        # model.summary(); quit(0)

        agent = AgentA2C(model)
        epi_num, epi_total_rewards, epi_end_rewards, epi_avg_rewards, epi_sim_times, t_sim_total, t_avg_sim_epi, t_real_total, t_avg_step, loss_total, loss_action, loss_value = agent.train(env, args.render, args.batch_size, args.num_updates)
        print("\nFinished training")
        # reward_test = agent.test(env, args.render)
        # print("Test Total Episode Reward: {}".format(reward_test))

        fmt = (util.print_time(t_real_total),util.print_time(t_sim_total),util.print_time(t_avg_sim_epi),t_avg_step,(t_sim_total/86400)/(t_real_total/3600),(t_sim_total/86400)/(t_real_total/86400))
        info = "runtime: {} real {} sim    | avg-time: {} sim-episode {:12.8f} real-step    |   {:.3f} sim-days/hour  {:.3f} sim-days/day".format(*fmt); print(info)
        argsinfo = "batch_size:{}   num_updates:{}   learning_rate:{}   loaded_model:{}".format(args.batch_size,args.num_updates,args.learning_rate, loaded_model); print(argsinfo)

        if args.plot_results and epi_num > 1:
            name = model_name+time.strftime("-%Y_%m_%d-%H-%M")
            xrng = np.arange(0, epi_num, 1)
            plt.figure(num=name, figsize=(24, 16), tight_layout=True); ax = []

            ax.insert(0, plt.subplot2grid((7, 1), (6, 0), rowspan=1))
            plt.plot(xrng, epi_sim_times[::1], alpha=1.0, label='Sim Time')
            ax[0].set_ylim(0,64)
            plt.xlabel('Episode'); plt.ylabel('Minutes'); plt.legend(loc='upper left')

            ax.insert(0, plt.subplot2grid((7, 1), (5, 0), rowspan=1, sharex=ax[-1]))
            plt.plot(xrng, loss_total[::1], alpha=0.7, label='Total Loss')
            plt.plot(xrng, loss_action[::1], alpha=0.7, label='Action Loss')
            plt.plot(xrng, loss_value[::1], alpha=0.7, label='Value Loss')
            # ax[0].set_ylim(0,60)
            plt.grid(axis='y',alpha=0.5)
            plt.xlabel('Episode'); plt.ylabel('Values'); plt.legend(loc='upper left')


            ax.insert(0, plt.subplot2grid((7, 1), (0, 0), rowspan=5, sharex=ax[-1]))

            plt.plot(xrng, epi_total_rewards[::1], alpha=0.4, label='Total Reward')
            epi_total_rewards_ema = talib.EMA(epi_total_rewards, timeperiod=epi_num//10+2)
            plt.plot(xrng, epi_total_rewards_ema, alpha=0.7, label='Total Reward EMA')

            # plt.plot(xrng, epi_avg_rewards[::1], alpha=0.2, label='Avg Reward')
            # plt.plot(xrng, epi_end_rewards[::1], alpha=0.5, label='Final Reward')
            # epi_end_rewards_ema = talib.EMA(epi_end_rewards, timeperiod=epi_num//10+2)
            # plt.plot(xrng, epi_end_rewards_ema, alpha=0.8, label='Final Reward EMA')
            # ax[0].set_ylim(0,30000)
            plt.grid(axis='y',alpha=0.5)
            plt.xlabel('Episode'); plt.ylabel('USD'); plt.legend(loc='upper left');

            plt.title(name+"    "+argsinfo+"\n"+info); plt.show()

        model.save_weights(model_file)
        print("SAVED model weights to {}".format(model_file))
