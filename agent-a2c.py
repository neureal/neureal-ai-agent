import time, os
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
# tf.config.experimental.enable_mlir_graph_optimization()
# tf.random.set_seed(0)
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import talib
import gym, gym_trader

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


@tf.function
def fixinfnan(t):
    zero = tf.constant(0.0, dtype=tf.float64)
    isinf = tf.math.is_inf(t)
    isneg = tf.math.equal(tf.math.sign(t),-1.0)
    ispos = tf.math.logical_not(isneg)
    isninf, ispinf = tf.math.logical_and(isinf, isneg), tf.math.logical_and(isinf, ispos)
    t = tf.where(ispinf, zero, t) # inf = 0.0
    t = tf.where(tf.math.logical_or(tf.math.is_nan(t), isninf), tf.float64.min, t) # nan = tf.float32.min, -inf = tf.float32.min
    return t

class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, groups, eps=1e-5, axis=-1, name=None):
        # TODO make diff axis work
        super(EvoNormS0, self).__init__(name=name)
        self.groups, self.eps, self.axis = groups, eps, axis

    def build(self, input_shape):
        inlen = len(input_shape)
        shape = [1] * inlen
        shape[self.axis] = input_shape[self.axis]
        self.gamma = self.add_weight(name="gamma", shape=shape, initializer=tf.initializers.Ones())
        self.beta = self.add_weight(name="beta", shape=shape, initializer=tf.initializers.Zeros())
        self.v1 = self.add_weight(name="v1", shape=shape, initializer=tf.initializers.Ones())

        groups = min(input_shape[self.axis], self.groups)
        self.group_shape = input_shape.as_list()
        self.group_shape[self.axis] = input_shape[self.axis] // groups
        self.group_shape.insert(self.axis, groups)

        std_shape = list(range(1, inlen+self.axis))
        std_shape.append(inlen)
        self.std_shape = tf.TensorShape(std_shape)

    @tf.function
    def call(self, inputs, training=True):
        input_shape = tf.shape(inputs)
        self.group_shape[0] = input_shape[0]
        grouped_inputs = tf.reshape(inputs, self.group_shape)
        _, var = tf.nn.moments(grouped_inputs, self.std_shape, keepdims=True)
        std = tf.sqrt(var + self.eps)
        std = tf.broadcast_to(std, self.group_shape)
        group_std = tf.reshape(std, input_shape)

        return (inputs * tf.sigmoid(self.v1 * inputs)) / group_std * self.gamma + self.beta


class Model(tf.keras.Model):
    def __init__(self, env, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        super().__init__('mlp_policy')
        # Logits are unnormalized log probabilities.
        # self.layer_action_dense_in = tf.keras.layers.Dense(128, kernel_initializer='identity', activation='relu', name='action_dense_in') # kernel_initializer='identity' sucks ass lol
        # self.layer_action_deconv1d_logits_out = tf.keras.layers.Conv1DTranspose(self.params_size/2, 2, name='action_deconv1d_logits_out')
        # self.action_size = self.action_size['action_dist_pair'] + self.action_size['action_dist_percent'] # latent_size

        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma, self.value_c, self.entropy_c = tf.constant(gamma, dtype=tf.float64), tf.constant(value_c, dtype=tf.float64), tf.constant(entropy_c, dtype=tf.float64)

        self._optimizer = tf.keras.optimizers.Adam(lr=lr)

        self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 1, 1, True, 128, 64, 16
        # self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 2, 2, True, 256, 128, 16
        # self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 4, 4, True, 1024, 512, 32
        # self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 4, 4, True, 2048, 1024, 32
        # self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 6, 6, True, 2048, 1024, 32

        self.net_arch = "inD{}-{:02d}D{}-{:02d}LS{}-outD{}".format(inp, self.net_DNN, mid, self.net_LSTM, mid, "-Evo"+str(evo) if self.net_evo else "")
        self.layer_action_dense, self.layer_action_lstm, self.layer_value_dense, self.layer_value_lstm = [], [], [], []

        ## action network
        if not self.net_evo:
            self.layer_action_dense_in = tf.keras.layers.Dense(inp, activation='relu', name='action_dense_in')
            for i in range(self.net_DNN): self.layer_action_dense.append(tf.keras.layers.Dense(mid, activation='relu', name='action_dense_{:02d}'.format(i)))
            for i in range(self.net_LSTM): self.layer_action_lstm.append(tf.keras.layers.LSTM(mid, stateful=True, name='action_lstm_{:02d}'.format(i)))
        else:
            self.layer_action_dense_in = tf.keras.layers.Dense(inp, activation=EvoNormS0(evo), use_bias=False, name='action_dense_in')
            for i in range(self.net_DNN): self.layer_action_dense.append(tf.keras.layers.Dense(mid, activation=EvoNormS0(evo), use_bias=False, name='action_dense_{:02d}'.format(i)))
            for i in range(self.net_LSTM): self.layer_action_lstm.append(tf.keras.layers.LSTM(mid, activation=EvoNormS0(evo), recurrent_activation=EvoNormS0(evo), use_bias=False, stateful=True, name='action_lstm_{:02d}'.format(i)))

        # self.params_size, self.action_size = env.action_space.n, 1 # Categorical
        self.num_components, self.action_size = 16, env.action_space.shape[0]
        self.params_size = tfp.layers.MixtureSameFamily.params_size(self.num_components, component_params_size=tfp.layers.MultivariateNormalTriL.params_size(self.action_size))
        self.layer_action_dense_logits_out = tf.keras.layers.Dense(self.params_size, name='action_dense_logits_out')
        # self.layer_action_de_conv1d_logits_out = tf.keras.layers.Conv1DTranspose(self.params_size/4, 4, name='action_de_conv1d_logits_out')
        self.layer_action_dist = tfp.layers.MixtureSameFamily(self.num_components, tfp.layers.MultivariateNormalTriL(self.action_size))

        ## value network
        if not self.net_evo:
            self.layer_value_dense_in = tf.keras.layers.Dense(inp, activation='relu', name='value_dense_in')
            for i in range(self.net_DNN): self.layer_value_dense.append(tf.keras.layers.Dense(mid, activation='relu', name='value_dense_{:02d}'.format(i)))
            for i in range(self.net_LSTM): self.layer_value_lstm.append(tf.keras.layers.LSTM(mid, stateful=True, name='value_lstm_{:02d}'.format(i)))
        else:
            self.layer_value_dense_in = tf.keras.layers.Dense(inp, use_bias=False, name='value_dense_in')
            for i in range(self.net_DNN): self.layer_value_dense.append(tf.keras.layers.Dense(mid, activation=EvoNormS0(evo), use_bias=False, name='value_dense_{:02d}'.format(i)))
            for i in range(self.net_LSTM): self.layer_value_lstm.append(tf.keras.layers.LSTM(mid, activation=EvoNormS0(evo), recurrent_activation=EvoNormS0(evo), use_bias=False, stateful=True, name='value_lstm_{:02d}'.format(i)))
        self.layer_value_dense_out = tf.keras.layers.Dense(1, name='value_dense_out')
        

        # pre build model
        sample = tf.expand_dims(tf.convert_to_tensor(env.observation_space.sample(), dtype=tf.float64), 0)
        action, value = self(sample)

    @tf.function
    def call(self, inputs, training=None):
        ## action network
        action = self.layer_action_dense_in(inputs)
        for i in range(self.net_DNN): action = self.layer_action_dense[i](action)
        for i in range(self.net_LSTM): action = self.layer_action_lstm[i](tf.expand_dims(action, axis=1))
        action = self.layer_action_dense_logits_out(action)
        # action = self.layer_action_de_conv1d_logits_out(action)

        ## value network
        value = self.layer_value_dense_in(inputs)
        for i in range(self.net_DNN): value = self.layer_value_dense[i](value)
        for i in range(self.net_LSTM): value = self.layer_value_lstm[i](tf.expand_dims(value, axis=1))
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
        return returns, advantages

    def _loss_action(self, actions, advantages, action_logits): # targets, output (acts_and_advs, layer_action_dense_logits_out)
        # dist = tfp.distributions.Categorical(logits=action_logits)
        # loss = -dist.log_prob(tf.squeeze(actions, axis=-1)) # cross_entropy
        # entropy = dist.entropy()

        dist = self.layer_action_dist(action_logits)
        actions = tf.cast(actions, dtype=tf.float64) # some envs have float32 actions
        # loss = -fixinfnan(dist.log_prob(actions)) # cross_entropy
        loss = dist.log_prob(actions)
        loss = -fixinfnan(loss) # cross_entropy
        # entropy = tf.reduce_mean(-dist.log_prob(dist.sample(4096)), axis=0)
        entropy = dist.sample(self.num_components)
        entropy = -dist.log_prob(entropy)
        entropy = tf.reduce_mean(entropy, axis=0)

        loss = tf.math.multiply(loss, advantages) # sample_weight
        # loss = tf.math.reduce_mean(loss) # tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        # loss = loss # tf.keras.losses.Reduction.NONE


        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        loss_action = loss - entropy * self.entropy_c
        return loss_action # shape = (batch size,)

    def _loss_value(self, returns, values): # targets, output (returns, layer_value_dense_out)
        # Value loss is typically MSE between value estimates and returns.
        returns = tf.expand_dims(returns, 1)
        loss_value = tf.keras.losses.mean_squared_error(returns, values) # regress [layer_value_dense_out] to [7,6,5,4,3,2,1,0]
        loss_value = loss_value * self.value_c
        return loss_value # shape = (batch size,)

    @tf.function
    def train(self, inputs, actions, advantages, returns, dones):
        inputs = tf.cast(inputs, dtype=tf.float64) # some envs have float32 observations
        # with tf.GradientTape() as tape:
        #     action_logits, value = self(inputs, training=True)
        #     loss_action = self._loss_action(actions, advantages, action_logits)
        #     loss_value = self._loss_value(returns, value)
        #     loss_total = loss_action + loss_value
        # gradients = tape.gradient(loss_total, self.trainable_variables)
        # self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # dones = tf.cast(dones, dtype=tf.bool)
        # loop through timesteps instead of batch so can keep batch size = 1 with LSTM and take out return_sequences=True, add self.reset_states() on episode end
        # TODO save state and reset state here, then restore state before leaving
        batch_size, event_shape = inputs.shape
        for i in tf.range(batch_size):
            with tf.GradientTape() as tape:
                action_logits, value = self(inputs[None, i], training=True)
                loss_action = self._loss_action(actions[None, i], advantages[None, i], action_logits)
                loss_value = self._loss_value(returns[None, i], value)
                loss_total = loss_action + loss_value
            gradients = tape.gradient(loss_total, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            if dones[i]: self.reset_states()

        return tf.reduce_mean(loss_total), tf.reduce_mean(loss_action), tf.reduce_mean(loss_value)


    @tf.function
    def _action_value(self, inputs):
        action_logits, value = self(inputs)
        # dist = tfp.distributions.Categorical(logits=action_logits)
        dist = self.layer_action_dist(action_logits)
        action = dist.sample()

        action, value = tf.squeeze(action), tf.squeeze(value) # get rid of single batch
        return action, value

    def action_value(self, obs):
        obs = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float64), 0) # add single batch
        # obs = tf.convert_to_tensor(obs[None, :], dtype=tf.float64))
        action, value = self._action_value(obs)
        action, value = action.numpy(), value.numpy()
        # print("action {} value {}".format(action, value))
        return action, value


class A2CAgent:
    def __init__(self, model):
        self.model = model

    def train(self, env, render=False, batch_sz=64, updates=250):
        # Storage helpers for a single batch of data.
        observations = np.empty((batch_sz, env.observation_space.shape[0]), dtype=env.observation_space.dtype)
        actions = np.empty((batch_sz, self.model.action_size), dtype=env.action_space.dtype)
        values = np.empty((batch_sz,), dtype=np.float64)
        rewards = np.empty((batch_sz,), dtype=np.float64)
        dones = np.empty((batch_sz,), dtype=np.bool)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        next_obs = env.reset()
        if render: env.render()

        loss_total, loss_action, loss_value, loss_total_cur, loss_action_cur, loss_value_cur = [], [], [], 0.0, 0.0, 0.0
        epi_total_rewards, epi_steps, epi_avg_reward, epi_end_reward, epi_avg_rewards, epi_end_rewards, epi_sim_times = [0.0], 0, 0, 0, [], [], []
        update, finished, early_quit = 0, False, False
        t_sim_total, t_sim_epi_start, t_real_start = 0.0, next_obs[0], time.time()
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
                    t_sim_epi = next_obs[0] - t_sim_epi_start
                    epi_sim_times.append(t_sim_epi / 60)
                    loss_total.append(loss_total_cur); loss_action.append(loss_action_cur); loss_value.append(loss_value_cur)
                    print("DONE episode #{:03d}  {} sim-epi-time {:10.2f} total-reward {:10.2f} avg-reward {:10.2f} end-reward".format(len(epi_total_rewards)-1, _print_time(t_sim_epi), epi_total_rewards[-1], epi_avg_reward, epi_end_reward))
                    self.model.reset_states()
                    # TODO train/update after every done
                    if update >= updates-1: finished = True; break
                    epi_total_rewards.append(0.0)
                    epi_steps = 0
                    t_sim_total += t_sim_epi

                    next_obs = env.reset()
                    if render: env.render()
                    t_sim_epi_start = next_obs[0]
            if early_quit: break

            _, next_value = self.model.action_value(next_obs)
            # returns ~ [7,6,5,4,3,2,1,0] for reward = 1 per step, advantages = returns - values output
            returns, advantages = self.model.calc_returns_advantages(rewards, dones, values, next_value)

            loss_total_cur, loss_action_cur, loss_value_cur = self.model.train(observations, actions, advantages, returns, dones)
            loss_total_cur, loss_action_cur, loss_value_cur = loss_total_cur.numpy(), loss_action_cur.numpy(), loss_value_cur.numpy()

            print("---> update [{:03d}/{:03d}]                                                                                            {:16.8f} loss_total {:16.8f} loss_action {:16.8f} loss_value".format(update, updates, loss_total_cur, loss_action_cur, loss_value_cur))
            # print("---> update [{:03d}/{:03d}]  {:10.2f} total-reward {:10.2f} avg-reward {:10.2f} last-reward  {:16.8f} loss_total {:16.8f} loss_action {:16.8f} loss_value".format(update, updates, epi_total_rewards[-1], epi_avg_reward, np.expm1(rewards[step]), loss_total_cur, loss_action_cur, loss_value_cur))
            update += 1

        epi_num = len(epi_end_rewards)
        t_sim_total += next_obs[0] - t_sim_epi_start
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



def _print_time(t):
    days=int(t//86400);hours=int((t-days*86400)//3600);mins=int((t-days*86400-hours*3600)//60);secs=int((t-days*86400-hours*3600-mins*60))
    return "{:4d}:{:02d}:{:02d}:{:02d}".format(days,hours,mins,secs)

class Args(): pass
args = Args()
args.batch_size = 1024 # about 1.5 hrs @ 1000.0 speed
args.num_updates = 100 # roughly batch_size * num_updates = total steps, unless last episode is long
args.learning_rate = 1e-4 # start with 4 for rough train, 5 for fine tune and 6 for when trained
args.render = False
args.plot_results = True

machine, device = 'dev', 0

if __name__ == '__main__':
    # env, model_name = gym.make('CartPole-v0'), "gym-A2C-CartPole" # Box(4,)	Discrete(2)	(-inf, inf)	200	100	195.0
    # env, model_name = gym.make('LunarLander-v2'), "gym-A2C-LunarLander" # Box(8,)	Discrete(4)	(-inf, inf)	1000	100	200
    env, model_name = gym.make('LunarLanderContinuous-v2'), "gym-A2C-LunarLanderCont" # Box(8,)	Box(2,)	(-inf, inf)	1000	100	200
    # env, model_name = gym.make('CarRacing-v0'), "gym-A2C-CarRacing" # Box(96, 96, 3)	Box(3,)	(-inf, inf)	1000	100	900
    # env, model_name = gym.make('Trader-v0', agent_id=device, env=2, speed=200.0), "gym-A2C-Trader2"

    with tf.device('/device:GPU:'+str(device)):
        # model = Model(num_actions=env.action_space.n)
        model = Model(env, lr=args.learning_rate);
        model_name += "-{}-{}-a{}".format(model.net_arch, machine, device)

        model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
        if tf.io.gfile.exists(model_file):
            model.load_weights(model_file, by_name=True, skip_mismatch=True)
            print("LOADED model weights from {}".format(model_file)); loaded_model = True
        # model.summary(); quit(0)

        agent = A2CAgent(model)
        epi_num, epi_total_rewards, epi_end_rewards, epi_avg_rewards, epi_sim_times, t_sim_total, t_avg_sim_epi, t_real_total, t_avg_step, loss_total, loss_action, loss_value = agent.train(env, args.render, args.batch_size, args.num_updates)
        print("\nFinished training")
        # reward_test = agent.test(env, args.render)
        # print("Test Total Episode Reward: {}".format(reward_test))

        fmt = (_print_time(t_real_total),_print_time(t_sim_total),_print_time(t_avg_sim_epi),t_avg_step,(t_sim_total/86400)/(t_real_total/3600),(t_sim_total/86400)/(t_real_total/86400))
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
