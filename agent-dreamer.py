import threading, time, os
import numpy as np
# np.set_printoptions(precision=8, suppress=True)
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import gym_trader
tf.config.experimental_run_functions_eagerly(True)
curdir = os.path.expanduser("~")

# maybe use MS-COCO to train both images and text captions, https://www.tensorflow.org/tutorials/text/image_captioning

# class RandomAgent(object):
#     def __init__(self, env):
#         self.env = env
#     def get_action(self, obs):
#         obs = gym.spaces.flatten(self.env.observation_space, obs)
#         print("agent: observation {} shape {} dtype {}\n{}".format(type(obs), obs.shape, obs.dtype, obs))
#         return env.action_space.sample()


class DreamerModel(tf.keras.Model):
    def __init__(self, env):
        super(DreamerModel, self).__init__()

        self.layer_action_dense_in = tf.keras.layers.Dense(128, activation='relu', name='action_dense_in')
        self.layer_action_dense_logits_out = tf.keras.layers.Dense(env.action_space['001_pair'].n, activation='linear', name='action_dense_logits_out')
        self.layer_action_dist_out = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Categorical(logits=input), name='action_dist_out')

        # self.layer_value_dense_in = tf.keras.layers.Dense(128, activation='relu', name='value_dense_in')
        # self.layer_value_dense_out = tf.keras.layers.Dense(1, activation='linear', name='value_dense_out')

    @tf.function
    def call(self, inputs, training=False):
        outputs = {}

        action = self.layer_action_dense_in(inputs['input'])
        action = self.layer_action_dense_logits_out(action)
        if training: outputs['action_logits'] = action
        outputs['action_dist'] = self.layer_action_dist_out(action)

        # value = self.layer_value_dense_in(inputs['input'])
        # outputs['value'] = self.layer_value_dense_out(value)

        return outputs

    _scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def _loss_discrete(self, targets, predicted):
        return self._scce(targets, predicted)
    # def _loss_continuous(self, targets, predicted):
    # 	return _scce(targets, predicted)

    _optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07)

    @tf.function
    def train(self, inputs):
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss = self._loss_discrete(tf.constant([2]), outputs['action_logits'])
            # loss += self._loss_continuous(targets['target_percent'], outputs['percent'])
        gradients = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return outputs, loss


# TODO use A2C agent as base
class ModelAgent(object):
    def __init__(self, model, env):
        self.model, self.env = model, env
    def get_action(self, obs):
        obs = gym.spaces.flatten(self.env.observation_space, obs)
        # print("agent: observation {} shape {} dtype {}\n{}".format(type(obs), obs.shape, obs.dtype, obs))

        # TODO loop through and send in seperate items in observation
        # data = {'six': gym.spaces.Discrete(6), 'bin': gym.spaces.MultiBinary(6)} obs
        input_buffer = {'input': tf.convert_to_tensor(obs[np.newaxis,...])}

        outputs, loss = self.model.train(input_buffer)

        action = env.action_space.sample()
        action['001_pair'] = int(outputs['action_dist'][0])

        return action


if __name__ == '__main__':
    me = 0
    # env = gym.make('FrozenLake-v0')
    # env = gym.make('CartPole-v0')
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    env = gym.make('Trader-v0', agent_id=me)
    env.seed(0)

    model = DreamerModel(env)
    obs = gym.spaces.flatten(env.observation_space, env.observation_space.sample())
    model({'input': tf.convert_to_tensor(obs[np.newaxis,...])})
    
    model_name = "gym-trader-test"
    model_file = "{}/tf_models/{}.h5".format(curdir, model_name)
    if tf.io.gfile.exists(model_file):
        model.load_weights(model_file, by_name=True)
        print("LOADED model weights from {}".format(model_file))

    agent = ModelAgent(model, env)
    for i_episode in range(2):
        reward_total = 0.0
        obs = env.reset()
        print("{}\n".format(obs))
        # env.render()
        for t_timesteps in range(3):
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            reward_total += reward

            print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))
            # env.render()
            time.sleep(1.01)

            if done: break
        # model.save(model_file)
        print("agent: episode {}{} | timesteps {} | reward mean {} total {}\n".format(i_episode+1, (' DONE' if done else ''), t_timesteps+1, reward_total/(t_timesteps+1), reward_total))

    env.close()

    model.summary()
    model.save_weights(model_file)
