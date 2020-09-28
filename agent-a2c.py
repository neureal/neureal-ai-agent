import time, os
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
import matplotlib.pyplot as plt
import gym
import gym_trader

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        sample = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        return sample


class Model(tf.keras.Model):
    def __init__(self, env):
        super().__init__('mlp_policy')
        # input_shape = [1] + list(num_obs)

        # self.layer_action_dense_in = tf.keras.layers.Dense(1024, activation='relu', input_shape=num_obs)
        self.layer_action_dense_in = tf.keras.layers.Dense(2048, kernel_initializer='identity', activation='relu', name='action_dense_in')
        self.layer_action_dense_01 = tf.keras.layers.Dense(1024, activation='relu', name='action_dense_01')
        self.layer_action_dense_02 = tf.keras.layers.Dense(1024, activation='relu', name='action_dense_02')
        self.layer_action_dense_03 = tf.keras.layers.Dense(1024, activation='relu', name='action_dense_03')
        self.layer_action_lstm_01 = tf.keras.layers.LSTM(1024, name='action_lstm_01')
        self.layer_action_lstm_02 = tf.keras.layers.LSTM(1024, name='action_lstm_02')
        self.layer_action_dense_logits_out = tf.keras.layers.Dense(env.action_space.n, kernel_initializer='identity', activation='linear', name='action_dense_logits_out') # Logits are unnormalized log probabilities.
        self.layer_action_sample = ProbabilityDistribution()

        # self.layer_value_dense_in = tf.keras.layers.Dense(1024, activation='relu', input_shape=num_obs)
        self.layer_value_dense_in = tf.keras.layers.Dense(2048, kernel_initializer='identity', activation='relu', name='value_dense_in')
        self.layer_value_dense_01 = tf.keras.layers.Dense(1024, activation='relu', name='value_dense_01')
        self.layer_value_dense_02 = tf.keras.layers.Dense(1024, activation='relu', name='value_dense_02')
        self.layer_value_dense_03 = tf.keras.layers.Dense(1024, activation='relu', name='value_dense_03')
        self.layer_value_lstm_01 = tf.keras.layers.LSTM(1024, name='value_lstm_01')
        self.layer_value_lstm_02 = tf.keras.layers.LSTM(1024, name='value_lstm_02')
        self.layer_value_dense_out = tf.keras.layers.Dense(1, kernel_initializer='identity', activation='linear', name='value_dense_out')
        
        self(tf.expand_dims(tf.convert_to_tensor(env.observation_space.sample()), 0))
        # self(tf.convert_to_tensor(env.observation_space.sample()[None, :]))

    @tf.function
    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        # input = tf.convert_to_tensor(inputs)

        # Separate hidden layers from the same input tensor.
        action = self.layer_action_dense_in(inputs) # seperate action model
        action = self.layer_action_dense_01(action)
        action = self.layer_action_dense_02(action)
        action = self.layer_action_dense_03(action)
        action = self.layer_action_lstm_01(tf.expand_dims(action, axis=1))
        action = self.layer_action_lstm_02(tf.expand_dims(action, axis=1))
        action = self.layer_action_dense_logits_out(action)

        value = self.layer_value_dense_in(inputs) # seperate value model
        value = self.layer_value_dense_01(value)
        value = self.layer_value_dense_02(value)
        value = self.layer_value_dense_03(value)
        value = self.layer_value_lstm_01(tf.expand_dims(value, axis=1))
        value = self.layer_value_lstm_02(tf.expand_dims(value, axis=1))
        value = self.layer_value_dense_out(value)

        return action, value

    # @tf.function
    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.layer_action_sample.predict_on_batch(logits)
        # Another way to sample actions:
        #     action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        action = tf.squeeze(action).numpy()
        value = tf.squeeze(value).numpy()
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
            optimizer=tf.keras.optimizers.RMSprop(lr=lr),
            # Define separate losses for policy logits and value estimate.
            loss=[self._action_logits_loss, self._value_loss]
        )
    
    def _print_time(self, t, t_real, avg_step):
        t_real = time.time() - t_real
        days=int(t//86400);hours=int((t-days*86400)//3600);mins=int((t-days*86400-hours*3600)//60);secs=int((t-days*86400-hours*3600-mins*60))
        print("total runtime (days) {:4d}:{:02d}:{:02d}:{:02d}   days/hour {:.8f} days/day {:.8f} avg step {:.8f}".format(days,hours,mins,secs, (t/86400) / (t_real/3600), (t/86400) / (t_real/86400), avg_step))

    def train(self, env, batch_sz=64, updates=250):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards, ep_steps, ep_end_balances = [0.0], 0, []
        next_obs = env.reset()
        env.render()
        update, finished, early_quit = 0, False, False
        t_sim_total, t_sim_start, t_real_start = 0.0, next_obs[0], time.time()
        steps_total, t1_time = 0, 0.0

        while update < updates or not finished:
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(tf.expand_dims(tf.convert_to_tensor(next_obs),0))
                # actions[step], values[step] = self.model.action_value(tf.convert_to_tensor(next_obs[None, :]))
                
                t1_start = time.perf_counter_ns()
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                t1_time += (time.perf_counter_ns() - t1_start) / 1e9 # seconds
                steps_total += 1
                if env.render(): early_quit = True; break

                ep_rewards[-1] += rewards[step]
                ep_steps += 1
                ep_avg_reward = np.expm1(ep_rewards[-1] / ep_steps)
                if dones[step]:
                    ep_end_balances.append(np.expm1(rewards[step]))
                    print("DONE episode #{:03d}  avg reward {:.2f}  end balance {:.2f}\n".format(len(ep_rewards)-1, ep_avg_reward, ep_end_balances[-1]))
                    if update >= updates-1: finished = True; break
                    ep_rewards.append(0.0)
                    ep_steps = 0
                    t_sim_total += next_obs[0] - t_sim_start
                    next_obs = env.reset()
                    t_sim_start = next_obs[0]
            if early_quit: break

            _, next_value = self.model.action_value(tf.expand_dims(tf.convert_to_tensor(next_obs),0))
            # _, next_value = self.model.action_value(tf.convert_to_tensor(next_obs[None, :]))
            # returns ~ [7,6,5,4,3,2,1,0] for reward = 1 per step, advs = returns - values output
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)

            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns]) # input, targets
            # print("update [{:03d}/{:03d}]  {} = {}".format(update + 1, updates, self.model.metrics_names, losses))
            # print("update [{:03d}/{:03d}]  avg reward {:.2f}  last balance {:.2f}  losses {}".format(update, updates, ep_avg_reward, np.expm1(rewards[step]), losses))
            update += 1

        t_sim_total += next_obs[0] - t_sim_start
        self._print_time(t_sim_total, t_real_start, t1_time/steps_total)
        return ep_end_balances

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        if render: env.render()
        while not done:
            action, _ = self.model.action_value(tf.expand_dims(tf.convert_to_tensor(obs),0))
            # action, _ = self.model.action_value(tf.convert_to_tensor(obs[None, :]))
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


class Args(): pass
args = Args()
args.batch_size = 512 # about 1.5 hrs @ 1000.0 speed
args.num_updates = 50 # routhly batch_size * num_updates = total steps, unless last episode is long
args.learning_rate = 7e-4 # start with -4 for rough train, -5 for fine tune and -6 for when trained
args.render_test = False
args.plot_results = True

machine, device = 'dev', '0'

if __name__ == '__main__':
    # env, model_name = gym.make('CartPole-v0'), "gym-A2C-CartPole" # Box(4,)	Discrete(2)	(-inf, inf)	200	100	195.0
    # env, model_name = gym.make('LunarLander-v2'), "gym-A2C-LunarLander" # Box(8,)	Discrete(4)	(-inf, inf)	1000	100	200
    # env = gym.make('LunarLanderContinuous-v2') # Box(8,)	Box(2,)	(-inf, inf)	1000	100	200
    # env = gym.make('CarRacing-v0') # Box(96, 96, 3)	Box(3,)	(-inf, inf)	1000	100	900
    env, model_name = gym.make('Trader-v0', env=3, speed=100.0), "gym-A2C-Trader2-"+device+"-"+machine

    with tf.device('/device:GPU:'+device):
        # model = Model(num_actions=env.action_space.n)
        model = Model(env)

        model_file = "{}/tf-data-models/{}.h5".format(curdir, model_name)
        if tf.io.gfile.exists(model_file):
            model.load_weights(model_file, by_name=True, skip_mismatch=True)
            print("LOADED model weights from {}".format(model_file))

        agent = A2CAgent(model, args.learning_rate)
        rewards_history = agent.train(env, args.batch_size, args.num_updates)
        print("Finished training")
        # reward_test = agent.test(env, args.render_test)
        # print("Test Total Episode Reward: {}".format(reward_test))

        if args.plot_results:
            name = model_name+time.strftime("-%Y_%m_%d-%H-%M")
            plt.figure(num=name, figsize=(24, 16), tight_layout=True)
            # plt.style.use('seaborn')
            plt.plot(np.arange(0, len(rewards_history), 1), rewards_history[::1], label='agent')
            plt.xlabel('Episode'); plt.ylabel('Final Balance')
            plt.title(name); plt.legend(loc='upper left')
            plt.show()

        # model.summary()
        model.save_weights(model_file)
        print("SAVED model weights to {}".format(model_file))
