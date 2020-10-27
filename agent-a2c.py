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
import matplotlib.pyplot as plt
import talib
import gym, gym_trader

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


# class DenseNormPReLU(tf.keras.layers.Layer):
#     def __init__(self, units, kernel_initializer='glorot_uniform', name=None):
#         super(DenseNormPReLU, self).__init__(name=name)
#         self.layer_dense = tf.keras.layers.Dense(units, kernel_initializer=kernel_initializer)
#         self.layer_norm = tf.keras.layers.BatchNormalization()
#         self.layer_activation = tf.keras.layers.PReLU()
#         # self.layer_norm = EvoNormS0(16, groups=8)
#     def call(self, inputs, training=None):
#         out = self.layer_dense(inputs)
#         out = self.layer_norm(out, training=training)
#         out = self.layer_activation(out)
#         return out


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


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        sample = tf.random.categorical(logits, 1)
        sample = tf.squeeze(sample, axis=-1)
        return sample


class Model(tf.keras.Model):
    def __init__(self, env):
        super().__init__('mlp_policy')
        # Logits are unnormalized log probabilities.
        # input_shape = [1] + list(num_obs)
        # self.layer_action_dense_in = tf.keras.layers.Dense(128, activation='relu', input_shape=num_obs)
        # self.layer_value_dense_in = tf.keras.layers.Dense(128, activation='relu', input_shape=num_obs)
        # self.layer_action_dense_in = tf.keras.layers.Dense(128, kernel_initializer='identity', activation='relu', name='action_dense_in') # kernel_initializer='identity' sucks ass lol
        # self.layer_action_deconv1d_logits_out = tf.keras.layers.Conv1DTranspose(self.params_size/2, 2, name='action_deconv1d_logits_out')
        self.params_size = env.action_space.n


        # self.layer_action_dense_in = DenseNormPReLU(2048, kernel_initializer='identity', name='action_dense_in')
        # self.layer_action_dense_01 = DenseNormPReLU(1024, name='action_dense_01')
        # self.layer_action_lstm_01 = tf.keras.layers.LSTM(1024, name='action_lstm_01')
        # self.layer_action_dense_logits_out = tf.keras.layers.Dense(self.params_size, kernel_initializer='identity', activation='linear', name='action_dense_logits_out')
        # self.model_action_sample = ProbabilityDistribution()
        # self.layer_value_dense_in = DenseNormPReLU(2048, kernel_initializer='identity', name='value_dense_in')
        # self.layer_value_dense_01 = DenseNormPReLU(1024, name='value_dense_01')
        # self.layer_value_lstm_01 = tf.keras.layers.LSTM(1024, name='value_lstm_01')
        # self.layer_value_dense_out = tf.keras.layers.Dense(1, kernel_initializer='identity', activation='linear', name='value_dense_out')



        # self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 1, 1, True, 128, 64, 16
        # self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 4, 4, True, 1024, 512, 32
        # self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 4, 4, True, 2048, 1024, 32
        self.net_DNN, self.net_LSTM, self.net_evo, inp, mid, evo = 6, 6, True, 2048, 1024, 32

        self.net_arch = "inD{}-{:02d}D{}-{:02d}LS{}-outD{}".format(inp, self.net_DNN, mid, self.net_LSTM, mid, "-Evo"+str(evo) if self.net_evo else "")
        self.layer_action_dense, self.layer_action_lstm, self.layer_value_dense, self.layer_value_lstm = [], [], [], []
        self.layer_action_dense_evo, self.layer_action_lstm_evo, self.layer_value_dense_evo, self.layer_value_lstm_evo = [], [], [], []

        ## action network
        if not self.net_evo:
            self.layer_action_dense_in = tf.keras.layers.Dense(inp, activation='relu', name='action_dense_in')
            for i in range(self.net_DNN): self.layer_action_dense.append(tf.keras.layers.Dense(mid, activation='relu', name='action_dense_{:02d}'.format(i)))
            for i in range(self.net_LSTM): self.layer_action_lstm.append(tf.keras.layers.LSTM(mid, name='action_lstm_{:02d}'.format(i)))
        else:
            self.layer_action_dense_in = tf.keras.layers.Dense(inp, use_bias=False, name='action_dense_in')
            self.layer_action_dense_in_evo = EvoNormS0(evo, name='action_dense_in_evo')
            for i in range(self.net_DNN):
                self.layer_action_dense.append(tf.keras.layers.Dense(mid, use_bias=False, name='action_dense_{:02d}'.format(i)))
                self.layer_action_dense_evo.append(EvoNormS0(evo, name='action_dense_{:02d}_evo'.format(i)))
            for i in range(self.net_LSTM):
                self.layer_action_lstm.append(tf.keras.layers.LSTM(mid, activation='linear', use_bias=False, name='action_lstm_{:02d}'.format(i)))
                self.layer_action_lstm_evo.append(EvoNormS0(evo, name='action_lstm_{:02d}_evo'.format(i)))
        self.layer_action_dense_logits_out = tf.keras.layers.Dense(self.params_size, activation='linear', name='action_dense_logits_out')
        self.model_action_sample = ProbabilityDistribution()

        ## value network
        if not self.net_evo:
            self.layer_value_dense_in = tf.keras.layers.Dense(inp, activation='relu', name='value_dense_in')
            for i in range(self.net_DNN): self.layer_value_dense.append(tf.keras.layers.Dense(mid, activation='relu', name='value_dense_{:02d}'.format(i)))
            for i in range(self.net_LSTM): self.layer_value_lstm.append(tf.keras.layers.LSTM(mid, name='value_lstm_{:02d}'.format(i)))
        else:
            self.layer_value_dense_in = tf.keras.layers.Dense(inp, use_bias=False, name='value_dense_in')
            self.layer_value_dense_in_evo = EvoNormS0(evo, name='value_dense_in_evo')
            for i in range(self.net_DNN):
                self.layer_value_dense.append(tf.keras.layers.Dense(mid, use_bias=False, name='value_dense_{:02d}'.format(i)))
                self.layer_value_dense_evo.append(EvoNormS0(evo, name='value_dense_{:02d}_evo'.format(i)))
            for i in range(self.net_LSTM):
                self.layer_value_lstm.append(tf.keras.layers.LSTM(mid, activation='linear', use_bias=False, name='value_lstm_{:02d}'.format(i)))
                self.layer_value_lstm_evo.append(EvoNormS0(evo, name='value_lstm_{:02d}_evo'.format(i)))
        self.layer_value_dense_out = tf.keras.layers.Dense(1, activation='linear', name='value_dense_out')
        

        # pre build model
        sample = tf.expand_dims(tf.convert_to_tensor(env.observation_space.sample(), dtype=tf.float64), 0)
        logits, value = self(sample)
        action = self.model_action_sample(logits)

    @tf.function
    def call(self, inputs, training=None):
        ## action network
        action = self.layer_action_dense_in(inputs)
        if self.net_evo: action = self.layer_action_dense_in_evo(action)
        for i in range(self.net_DNN):
            action = self.layer_action_dense[i](action)
            if self.net_evo: action = self.layer_action_dense_evo[i](action)
        for i in range(self.net_LSTM):
            action = self.layer_action_lstm[i](tf.expand_dims(action, axis=1))
            if self.net_evo: action = self.layer_action_lstm_evo[i](action)
        action = self.layer_action_dense_logits_out(action)

        ## value network
        value = self.layer_value_dense_in(inputs)
        if self.net_evo: value = self.layer_value_dense_in_evo(value)
        for i in range(self.net_DNN):
            value = self.layer_value_dense[i](value)
            if self.net_evo: value = self.layer_value_dense_evo[i](value)
        for i in range(self.net_LSTM):
            value = self.layer_value_lstm[i](tf.expand_dims(value, axis=1))
            if self.net_evo: value = self.layer_value_lstm_evo[i](value)
        value = self.layer_value_dense_out(value)

        return action, value

    # @tf.function
    def _action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.model_action_sample.predict_on_batch(logits)
        # Another way to sample actions:
        #     action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        action = tf.squeeze(action)
        value = tf.squeeze(value)
        return action, value

    def action_value(self, obs):
        obs = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float64), 0) # add batch size 1
        # obs = tf.convert_to_tensor(obs[None, :], dtype=tf.float64))
        action, value = self._action_value(obs)
        action, value = action.numpy(), value.numpy()
        # print("action {} value {}".format(action, value))
        return action, value


class A2CAgent:
    def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c

        self.model = model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            # Define separate losses for policy logits and value estimate.
            loss=[self._action_logits_loss, self._value_loss]
        )

    def train(self, env, render=False, batch_sz=64, updates=250):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        next_obs = env.reset()
        if render: env.render()

        loss_total, loss_action, loss_value, losses = [], [], [], (0.0,0.0,0.0)
        epi_total_rewards, epi_steps, epi_avg_reward, epi_end_reward, epi_avg_rewards, epi_end_rewards, epi_sim_times = [0.0], 0, 0, 0, [], [], []
        update, finished, early_quit = 0, False, False
        t_sim_total, t_sim_epi_start, t_real_start = 0.0, next_obs[0], time.time()
        steps_total, t_steps_total = 0, 0.0

        while update < updates or not finished:
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs)
                
                step_time_start = time.perf_counter_ns()
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
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
                    loss_total.append(losses[0]); loss_action.append(losses[1]); loss_value.append(losses[2])
                    print("DONE episode #{:03d}  sim-epi-time {}    total-reward {:.2f}    avg-reward {:.2f}    end-reward {:.2f}\n".format(len(epi_total_rewards)-1, _print_time(t_sim_epi), epi_total_rewards[-1], epi_avg_reward, epi_end_reward))
                    if update >= updates-1: finished = True; break
                    epi_total_rewards.append(0.0)
                    epi_steps = 0
                    t_sim_total += t_sim_epi

                    next_obs = env.reset()
                    if render: env.render()
                    t_sim_epi_start = next_obs[0]
            if early_quit: break

            _, next_value = self.model.action_value(next_obs)
            # returns ~ [7,6,5,4,3,2,1,0] for reward = 1 per step, advs = returns - values output
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)

            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns]) # input, targets
            # print("update [{:03d}/{:03d}]  {} = {}".format(update + 1, updates, self.model.metrics_names, losses))
            print("update [{:03d}/{:03d}]  total-reward {:.2f}   avg-reward {:.2f}   last-reward {:.2f}   losses {}".format(update, updates, epi_total_rewards[-1], epi_avg_reward, np.expm1(rewards[step]), losses))
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

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        # test = np.asarray(next_value)
        returns = np.append(np.zeros_like(rewards), [next_value], axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        return returns, advantages

    def _action_logits_loss(self, actions_and_advantages, logits): # targets, output (acts_and_advs, layer_action_dense_logits_out)
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        # policy_loss = weighted_sparse_ce(actions, logits)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages) # actions = samples, logits = distribution # try to  # return scaler

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs) # return batch size array # wtf is this

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        total_loss = policy_loss - self.entropy_c * entropy_loss
        return total_loss # batch size

    def _value_loss(self, returns, value): # targets, output (returns, layer_value_dense_out)
        # Value loss is typically MSE between value estimates and returns.
        value_loss = self.value_c * tf.keras.losses.mean_squared_error(returns, value) # regress [layer_value_dense_out] to [7,6,5,4,3,2,1,0]
        return value_loss # batch size


def _print_time(t):
    days=int(t//86400);hours=int((t-days*86400)//3600);mins=int((t-days*86400-hours*3600)//60);secs=int((t-days*86400-hours*3600-mins*60))
    return "{:4d}:{:02d}:{:02d}:{:02d}".format(days,hours,mins,secs)

class Args(): pass
args = Args()
args.batch_size = 2048 # about 1.5 hrs @ 1000.0 speed
args.num_updates = 10 # roughly batch_size * num_updates = total steps, unless last episode is long
args.learning_rate = 1e-6 # start with 4 for rough train, 5 for fine tune and 6 for when trained
args.render = False
args.plot_results = True

machine, device = 'dev', 0

if __name__ == '__main__':
    # env, model_name = gym.make('CartPole-v0'), "gym-A2C-CartPole" # Box(4,)	Discrete(2)	(-inf, inf)	200	100	195.0
    # env, model_name = gym.make('LunarLander-v2'), "gym-A2C-LunarLander" # Box(8,)	Discrete(4)	(-inf, inf)	1000	100	200
    # env, model_name = gym.make('LunarLanderContinuous-v2'), "gym-A2C-LunarLanderCont" # Box(8,)	Box(2,)	(-inf, inf)	1000	100	200
    # env, model_name = gym.make('CarRacing-v0'), "gym-A2C-CarRacing" # Box(96, 96, 3)	Box(3,)	(-inf, inf)	1000	100	900
    env, model_name = gym.make('Trader-v0', agent_id=device, env=2, speed=200.0), "gym-A2C-Trader2"

    with tf.device('/device:GPU:'+str(device)):
        # model = Model(num_actions=env.action_space.n)
        model = Model(env);
        model_name += "-{}-{}-a{}".format(model.net_arch, machine, device)

        model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
        if tf.io.gfile.exists(model_file):
            model.load_weights(model_file, by_name=True, skip_mismatch=True)
            print("LOADED model weights from {}".format(model_file)); loaded_model = True
        # model.summary(); quit(0)

        agent = A2CAgent(model, args.learning_rate)
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
            plt.figure(num=name, figsize=(24, 16), tight_layout=True)

            ax3 = plt.subplot2grid((7, 1), (6, 0), rowspan=1)
            plt.plot(xrng, epi_sim_times[::1], alpha=1.0, label='Sim Time')
            ax3.set_ylim(0,64)
            plt.xlabel('Episode'); plt.ylabel('Minutes'); plt.legend(loc='upper left')

            ax2 = plt.subplot2grid((7, 1), (5, 0), rowspan=1, sharex=ax3)
            plt.plot(xrng, loss_total[::1], alpha=0.7, label='Total Loss')
            plt.plot(xrng, loss_action[::1], alpha=0.7, label='Action Loss')
            plt.plot(xrng, loss_value[::1], alpha=0.7, label='Value Loss')
            # ax2.set_ylim(0,60)
            plt.xlabel('Episode'); plt.ylabel('Values'); plt.legend(loc='upper left')

            ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=5, sharex=ax3)
            # plt.plot(xrng, epi_total_rewards[::1], alpha=0.4, label='Total Reward')
            # epi_total_rewards_ema = talib.EMA(epi_total_rewards, timeperiod=epi_num//10+2)
            # plt.plot(xrng, epi_total_rewards_ema, alpha=0.7, label='Total Reward EMA')
            plt.plot(xrng, epi_avg_rewards[::1], alpha=0.2, label='Avg Reward')
            plt.plot(xrng, epi_end_rewards[::1], alpha=0.5, label='Final Reward')
            epi_end_rewards_ema = talib.EMA(epi_end_rewards, timeperiod=epi_num//10+2)
            plt.plot(xrng, epi_end_rewards_ema, alpha=0.8, label='Final Reward EMA')
            ax1.set_ylim(0,30000)
            plt.xlabel('Episode'); plt.ylabel('USD'); plt.legend(loc='upper left');

            plt.title(name+"    "+argsinfo+"\n"+info); plt.show()

        model.save_weights(model_file)
        print("SAVED model weights to {}".format(model_file))
