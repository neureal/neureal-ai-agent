import time, os, talib
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
import gym, gym_trader
import model_util as util

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


class ActorCriticAI(tf.keras.Model):
    def __init__(self, env, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        super().__init__('mlp_policy')
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma, self.value_c, self.entropy_c = tf.constant(gamma, dtype=self.compute_dtype), tf.constant(value_c, dtype=self.compute_dtype), tf.constant(entropy_c, dtype=self.compute_dtype)
        # self.float_max = tf.constant(tf.dtypes.as_dtype(self.compute_dtype).max, dtype=self.compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(tf.dtypes.as_dtype(self.compute_dtype).max), dtype=self.compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(self.compute_dtype).eps, dtype=self.compute_dtype)

        self._optimizer = tf.keras.optimizers.Adam(lr=lr)
        self._optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self._optimizer)


        self.action_num_components = 32
        self.action_params_size, self.action_is_discrete, self.action_dist = util.gym_action_dist(env.action_space, dtype=self.compute_dtype, num_components=self.action_num_components)

        self.obs_is_vec, self.obs_sample = util.gym_obs_embed(env.observation_space, dtype=self.compute_dtype)

        # self.net_DNN, self.net_LSTM, inp, mid, evo = 1, 0, 128, 64, 16
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 0, 1, 1024, 512, 16
        # self.net_DNN, self.net_LSTM, inp, mid, evo = 4, 1, 2048, 1024, 32
        self.net_DNN, self.net_LSTM, inp, mid, evo = 8, 1, 4096, 2048, 64
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
        outputs = self(self.obs_sample)
        self.reset_states() # needed because the previous call unnormally advances the state!

    @tf.function
    def call(self, inputs, training=None):
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
        # batch_sz = inputs.shape[0]
        # action = tf.expand_dims(action, axis=1)
        # action = self.layer_deconv1d_logits_out(action)
        # action = tf.reshape(action, (batch_sz, self.action_params_size))

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

        print("tracing -> call")
        return action, value

    # TODO convert to tf graph? or share this processing with CPU, use test-tf2-keras-a2c.py and new generic agent.py
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
        # advantages = returns - values
        # advantages = np.abs(advantages) # for MAE, could be np.square() for MSE
        # return returns, advantages
        return returns

    def _loss_action(self, action_logits, actions, advantages): # layer_action_dense_logits_out, targets, returns - values
        loss, entropy = util.dist_loss_entropy(self.action_dist, action_logits, actions, self.action_is_discrete, self.action_num_components)

        loss = tf.math.multiply(loss, advantages) # sample_weight
        # loss = loss - returns
        # loss = tf.math.square(loss)

        # # PPO
        # self.clip_pram = 0.2
        # loss_alt = tf.clip_by_value(loss, 1.0 - self.clip_pram, 1.0 + self.clip_pram)
        # loss_alt = tf.math.multiply(loss_alt, advantages)
        # loss = tf.math.minimum(loss, loss_alt)

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
        # loss_value = tf.math.abs(advantages) * self.value_c # TODO try this when works in tf
        return loss_value # shape = (batch size,)

    @tf.function
    # def train(self, inputs, actions, returns, advantages, dones):
    # def train(self, inputs, actions, returns, advantages):
    def _train(self, inputs, actions, returns):
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.float_eps))
        # returns = tf.math.sigmoid(returns)
        self.reset_states()
        with tf.GradientTape() as tape:
            action_logits, values = self(inputs, training=True) # redundant, but needed to be able to calculate the gradients backwards
            values = tf.squeeze(values, axis=-1)
            advantages = returns - values # should calculate returns inside gradient tape so can do backprop on it!!
            # advantages = tf.math.square(advantages)
            # advantages = tf.math.abs(advantages) # CRASHES!!
            # isneg = tf.math.equal(tf.math.sign(advantages),-1.0)
            # advantages = tf.where(isneg, advantages*-1.0, advantages)

            loss_action = self._loss_action(action_logits, actions, advantages)
            # loss_value = self._loss_value(returns, values) # redundant, but needed to be able to calculate the gradients backwards
            # loss_value = self._loss_value(returns, new_values, advantages)
            loss_value = self._loss_value(advantages)
            loss_total = loss_action + loss_value
        gradients = tape.gradient(loss_total, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.reset_states()

        print("tracing -> _train")
        return tf.reduce_mean(loss_total), tf.reduce_mean(loss_action), tf.reduce_mean(loss_value)

    def train(self, obs, actions, returns):
        obs = util.np_to_tf(obs, dtype=self.compute_dtype)
        actions = util.np_to_tf(actions)
        returns = tf.convert_to_tensor(returns, dtype=self.compute_dtype)
        loss_total_cur, loss_action_cur, loss_value_cur = self._train(obs, actions, returns)
        return loss_total_cur.numpy(), loss_action_cur.numpy(), loss_value_cur.numpy()


    @tf.function
    def _action_value(self, inputs):
        action_logits, value = self(inputs)
        action = util.dist_sample(self.action_dist, action_logits)
        print("tracing -> _action_value")
        return action, tf.squeeze(value)

    def action_value(self, obs):
        obs = tf.convert_to_tensor([obs], dtype=self.compute_dtype) # add single batch
        action, value = self._action_value(obs)
        action, value = util.tf_to_np(action), util.tf_to_np(value)
        # print("action {} value {}".format(action, value))
        return action, value


class AgentA2C:
    def __init__(self, model):
        self.model = model

    def train(self, env, render=False, batch_sz=64, updates=250):
        # Storage helpers for a single batch of data.
        actions = util.gym_action_get_mem(env.action_space, batch_sz)
        observations = util.gym_obs_get_mem(env.observation_space, batch_sz)
        values = np.zeros((batch_sz,), dtype=model.compute_dtype)
        rewards = np.zeros((batch_sz,), dtype=model.compute_dtype)
        dones = np.zeros((batch_sz,), dtype=np.bool)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        next_obs = util.gym_obs_get(env.observation_space, env.reset())
        if render: env.render()

        loss_total, loss_action, loss_value, loss_total_cur, loss_action_cur, loss_value_cur = [], [], [], 0.0, 0.0, 0.0
        epi_total_rewards, epi_steps, epi_avg_reward, epi_end_reward, epi_avg_rewards, epi_end_rewards = [0.0], 0, 0, 0, [], []
        update, finished, early_quit = 0, False, False
        t_total, t_epi_times, t_epi_start = 0.0, [], time.time()
        steps_total, t_steps_total = 0, 0.0

        while update < updates or not finished:
            for step in range(batch_sz):
                util.update_mem(observations, step, next_obs)
                action, values[step] = self.model.action_value(next_obs)
                util.update_mem(actions, step, action)
                
                step_time_start = time.perf_counter_ns()
                next_obs, rewards[step], dones[step], _ = env.step(action)
                next_obs = util.gym_obs_get(env.observation_space, next_obs)
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
                    t_epi = time.time() - t_epi_start
                    t_epi_times.append(t_epi / 60)
                    loss_total.append(loss_total_cur); loss_action.append(loss_action_cur); loss_value.append(loss_value_cur)
                    print("DONE episode #{:03d}  {} epi-time {:10.2f} total-reward {:10.2f} avg-reward {:10.2f} end-reward".format(len(epi_total_rewards)-1, util.print_time(t_epi), epi_total_rewards[-1], epi_avg_reward, epi_end_reward))
                    # TODO train/update after every done. wait till new generic agent.py
                    if update >= updates-1: finished = True; break
                    epi_total_rewards.append(0.0)
                    epi_steps = 0
                    t_total += t_epi

                    next_obs = util.gym_obs_get(env.observation_space, env.reset())
                    if render: env.render()
                    t_epi_start = time.time()
            if early_quit: break

            self.model.reset_states() # needed because otherwise this next call unnormally advances the state, ineffecient!
            _, next_value = self.model.action_value(next_obs)
            # returns ~ [7,6,5,4,3,2,1,0] for reward = 1 per step, advantages = returns - values output
            # returns, advantages = self.model.calc_returns_advantages(rewards, dones, values, next_value)
            returns = self.model.calc_returns_advantages(rewards, dones, values, next_value)

            # loss_total_cur, loss_action_cur, loss_value_cur = self.model.train(observations, actions, returns, advantages, dones)
            # loss_total_cur, loss_action_cur, loss_value_cur = self.model.train(observations, actions, returns, advantages)
            loss_total_cur, loss_action_cur, loss_value_cur = self.model.train(observations, actions, returns)

            print("---> update [{:03d}/{:03d}]                                                                                            {:16.8f} loss_total {:16.8f} loss_action {:16.8f} loss_value".format(update, updates, loss_total_cur, loss_action_cur, loss_value_cur))
            # print("---> update [{:03d}/{:03d}]  {:10.2f} total-reward {:10.2f} avg-reward {:10.2f} last-reward  {:16.8f} loss_total {:16.8f} loss_action {:16.8f} loss_value".format(update, updates, epi_total_rewards[-1], epi_avg_reward, np.expm1(rewards[step]), loss_total_cur, loss_action_cur, loss_value_cur))
            update += 1

        epi_num = len(epi_end_rewards)
        t_total += time.time() - t_epi_start
        t_avg_epi = (t_total / epi_num) if epi_num > 0 else 0.0
        t_avg_step = (t_steps_total / steps_total) if steps_total > 0 else 0.0

        return epi_num, np.asarray(epi_total_rewards), np.asarray(epi_end_rewards), np.asarray(epi_avg_rewards), np.asarray(t_epi_times), t_total, t_avg_epi, t_avg_step, np.asarray(loss_total), np.asarray(loss_action), np.asarray(loss_value)

    # def test(self, env, render=False):
    #     obs, done, ep_reward = env.reset(), False, 0
    #     if render: env.render()
    #     while not done:
    #         action, _ = self.model.action_value(obs)
    #         obs, reward, done, _ = env.step(action)
    #         if render: env.render()
    #         ep_reward += reward
    #     return ep_reward



class Args(): pass
args = Args()
args.num_updates = 500 # roughly batch_sz * num_updates = total steps, unless last episode is long
args.learning_rate = 1e-6 # start with 4 for rough train, 5 for fine tune and 6 for when trained
args.entropy_c = 1e-8 # scaler for entropy loss contribution
args.value_c = 1.0 # scaler for value loss contribution
args.render = True
args.plot_results = True

machine, device = 'dev', 0

trader, trader_env, trader_speed = False, 3, 180.0
# import envs_local.bipedal_walker as env_bipedal_walker
# import envs_local.car_racing as env_car_racing
if __name__ == '__main__':
    # env, model_name, batch_sz, glimt = gym.make('CartPole-v0'), "gym-A2C-CartPole", 256, 0.05                                       # Box((4),-inf:inf,float32)         Discrete(2,int64)             200    100  195.0 # obs return float64 even though specs say float32?!?
    # env, model_name, batch_sz, glimt = gym.make('MountainCar-v0'), "gym-A2C-MountainCar", 1024, 0.05                                # Box((2),-1.2:0.6,float32)         Discrete(3)
    # env, model_name, batch_sz, glimt = gym.make('MountainCarContinuous-v0'), "gym-A2C-MountainCarContinuous", 256, 0.05             # Box((2),-1.2:0.6,float32)         Box((1),-1.0:1.0,float32)
    # env, model_name, batch_sz, glimt = gym.make('LunarLander-v2'), "gym-A2C-LunarLander", 1024, 0.2                                 # Box((8),-inf:inf,float32)         Discrete(4,int64)             1000   100  200
    # env, model_name, batch_sz, glimt = gym.make('LunarLanderContinuous-v2'), "gym-A2C-LunarLanderCont", 1024, 0.2                   # Box((8),-inf:inf,float32)         Box((2),-1.0:1.0,float32)     1000   100  200
    # env, model_name, batch_sz, glimt = gym.make('BipedalWalker-v3'), "gym-A2C-BipedalWalker", 256, 1.0                              # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name, batch_sz, glimt = env_bipedal_walker.BipedalWalker(), "gym-A2C-BipedalWalker", 256, 1.0                        # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name, batch_sz, glimt = gym.make('BipedalWalkerHardcore-v3'), "gym-A2C-BipedalWalkerHardcore", 256, 1.0              # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name, batch_sz, glimt = env_bipedal_walker.BipedalWalkerHardcore(), "gym-A2C-BipedalWalkerHardcore", 256, 1.0        # Box((24),-inf:inf,float32)        Box((4),-1.0:1.0,float32)
    # env, model_name, batch_sz, glimt = gym.make('CarRacing-v0'), "gym-A2C-CarRacing", 256, 1.0                                      # Box((96,96,3),0:255,uint8)        Box((3),-1.0:1.0,float32)     1000   100  900 # MEMORY LEAK!
    # env, model_name, batch_sz, glimt = env_car_racing.CarRacing(), "gym-A2C-CarRacing", 256, 1.0                                    # Box((96,96,3),0:255,uint8)        Box((3),-1.0:1.0,float32)     1000   100  900 # MEMORY LEAK!
    # env, model_name, batch_sz, glimt = gym.make('QbertNoFrameskip-v4'), "gym-A2C-QbertNoFrameskip-v4", 256, 1.0                     # Box((210,160,3),0:255,uint8)      Discrete(6)                   400000 100  None
    # env, model_name, batch_sz, glimt = gym.make('BoxingNoFrameskip-v4'), "gym-A2C-BoxingNoFrameskip-v4", 256, 1.0                   # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name, batch_sz, glimt = gym.make('CentipedeNoFrameskip-v4'), "gym-A2C-CentipedeNoFrameskip-v4", 256, 1.0             # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name, batch_sz, glimt = gym.make('PitfallNoFrameskip-v4'), "gym-A2C-PitfallNoFrameskip-v4", 256, 1.0                 # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name, batch_sz, glimt = gym.make('MontezumaRevengeNoFrameskip-v4'), "gym-A2C-MontezumaRevenge-v4", 256, 1.0          # Box((210,160,3),0:255,uint8)      Discrete(18)                  400000 100  None
    # env, model_name, batch_sz, glimt = gym.make('Copy-v0'), "gym-A2C-Copy", 32, 0.01                                                # Discrete(6)                       Tuple(Dis(2),Dis(2),Dis(5))
    # env, model_name, batch_sz, glimt = gym.make('RepeatCopy-v0'), "gym-A2C-RepeatCopy", 32, 0.01                                    # Discrete(6)                       Tuple(Dis(2),Dis(2),Dis(5))
    # env, model_name, batch_sz, glimt = gym.make('ReversedAddition3-v0'), "gym-A2C-ReversedAddition3", 32, 0.01                      # Discrete(4)                       Tuple(Dis(4),Dis(2),Dis(3)) # can't solve???
    env, model_name, batch_sz, glimt, trader = gym.make('Trader-v0', agent_id=device, env=trader_env, speed=trader_speed), "gym-A2C-Trader2", 2048, 64.0, True

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
        epi_num, epi_total_rewards, epi_end_rewards, epi_avg_rewards, t_epi_times, t_total, t_avg_epi, t_avg_step, loss_total, loss_action, loss_value = agent.train(env, args.render, batch_sz, args.num_updates)
        print("\nFinished training")
        # reward_test = agent.test(env, args.render)
        # print("Test Total Episode Reward: {}".format(reward_test))

        fmt = (util.print_time(t_total),util.print_time(t_avg_epi),t_avg_step)
        info = "total runtime: {}    | avg-episode: {}    avg-step: {:12.8f}".format(*fmt); print(info)
        argsinfo = "batch_sz:{}   num_updates:{}   learning_rate:{}   entropy_c:{}   value_c:{}   loaded_model:{}".format(batch_sz,args.num_updates,args.learning_rate,args.entropy_c,args.value_c,loaded_model); print(argsinfo)

        if args.plot_results and epi_num > 1:
            name = model_name+time.strftime("-%Y_%m_%d-%H-%M")
            xrng = np.arange(0, epi_num, 1)
            plt.figure(num=name, figsize=(24, 16), tight_layout=True); ax = []

            ax.insert(0, plt.subplot2grid((7, 1), (6, 0), rowspan=1))
            plt.plot(xrng, t_epi_times[::1]*(trader_speed if trader else 1.0), alpha=1.0, label='Episode Time')
            ax[0].set_ylim(0,glimt)
            plt.xlabel('Episode'); plt.ylabel('Minutes'); plt.legend(loc='upper left')

            ax.insert(0, plt.subplot2grid((7, 1), (5, 0), rowspan=1, sharex=ax[-1]))
            plt.plot(xrng, loss_total[::1], alpha=0.7, label='Total Loss')
            plt.plot(xrng, loss_action[::1], alpha=0.7, label='Action Loss')
            plt.plot(xrng, loss_value[::1], alpha=0.7, label='Value Loss')
            # ax[0].set_ylim(0,60)
            plt.grid(axis='y',alpha=0.5)
            plt.xlabel('Episode'); plt.ylabel('Values'); plt.legend(loc='upper left')

            ax.insert(0, plt.subplot2grid((7, 1), (0, 0), rowspan=5, sharex=ax[-1]))
            if trader:
                plt.plot(xrng, epi_avg_rewards[::1], alpha=0.2, label='Avg Reward')
                plt.plot(xrng, epi_end_rewards[::1], alpha=0.5, label='Final Reward')
                epi_end_rewards_ema = talib.EMA(epi_end_rewards, timeperiod=epi_num//10+2)
                plt.plot(xrng, epi_end_rewards_ema, alpha=0.8, label='Final Reward EMA')
                ax[0].set_ylim(0,30000)
            else:
                plt.plot(xrng, epi_total_rewards[::1], alpha=0.4, label='Total Reward')
                epi_total_rewards_ema = talib.EMA(epi_total_rewards, timeperiod=epi_num//10+2)
                plt.plot(xrng, epi_total_rewards_ema, alpha=0.7, label='Total Reward EMA')
            plt.grid(axis='y',alpha=0.5)
            plt.xlabel('Episode'); plt.ylabel('USD'); plt.legend(loc='upper left');

            plt.title(name+"    "+argsinfo+"\n"+info); plt.show()

        model.save_weights(model_file)
        print("SAVED model weights to {}".format(model_file))
