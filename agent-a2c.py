import numpy as np
# np.set_printoptions(precision=8, suppress=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
tf.config.experimental_run_functions_eagerly(True)


class Args(): pass
args = Args()
args.batch_size = 64
args.num_updates = 150
args.learning_rate = 7e-3
args.render_test = True
args.plot_results = True


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')

        self.layer_action_dense_in = tf.keras.layers.Dense(128, activation='relu')
        self.layer_action_logits_out = tf.keras.layers.Dense(num_actions, activation='linear', name='policy_logits') # Logits are unnormalized log probabilities.
        self.layer_action_sample = ProbabilityDistribution()

        self.layer_value_dense_in = tf.keras.layers.Dense(128, activation='relu')
        self.layer_value_dense_out = tf.keras.layers.Dense(1, activation='linear', name='value')

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)

        # Separate hidden layers from the same input tensor.
        hidden_action = self.layer_action_dense_in(x) # seperate action model
        hidden_value = self.layer_value_dense_in(x) # seperate value model
        return self.layer_action_logits_out(hidden_action), self.layer_value_dense_out(hidden_value)

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.layer_action_sample.predict_on_batch(logits)
        # Another way to sample actions:
        #     action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


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

    def train(self, env, batch_sz=64, updates=250):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    print("DONE episode [{:03d}] total rewards [{}]".format(len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            # returns ~ [7,6,5,4,3,2,1,0] for reward = 1 per step, advs = returns - values output
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)

            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns]) # input, targets
            print("update [{:03d}/{:03d}] {} losses {}".format(update + 1, updates, self.model.metrics_names,losses))

        return ep_rewards

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render: env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        return returns, advantages

    def _action_logits_loss(self, actions_and_advantages, logits): # targets, output (acts_and_advs, layer_action_logits_out)
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
        if total_loss[0] < 0.0:
            print('test')
        return total_loss # batch size

    def _value_loss(self, returns, value): # targets, output (returns, layer_value_dense_out)
        # Value loss is typically MSE between value estimates and returns.
        value_loss = self.value_c * tf.keras.losses.mean_squared_error(returns, value) # regress [layer_value_dense_out] to [7,6,5,4,3,2,1,0]
        return value_loss # batch size



if __name__ == '__main__':
    # env = gym.make('CartPole-v0') # Box(4,)	Discrete(2)	(-inf, inf)	200	100	195.0
    env = gym.make('LunarLander-v2') # Box(8,)	Discrete(4)	(-inf, inf)	1000	100	200
    # env = gym.make('LunarLanderContinuous-v2') # Box(8,)	Box(2,)	(-inf, inf)	1000	100	200
    # env = gym.make('CarRacing-v0') # Box(96, 96, 3)	Box(3,)	(-inf, inf)	1000	100	900

    model = Model(num_actions=env.action_space.n)
    agent = A2CAgent(model, args.learning_rate)

    rewards_history = agent.train(env, args.batch_size, args.num_updates)
    print("Finished training. Testing...")
    reward_test = agent.test(env, args.render_test)
    print("Total Episode Reward: {} out of 200".format(reward_test))

    if args.plot_results:
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(rewards_history), 1), rewards_history[::1])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
