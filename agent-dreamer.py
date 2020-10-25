import threading, time, os
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
import gym
import gym_trader

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


# maybe use MS-COCO to train both images and text captions, https://www.tensorflow.org/tutorials/text/image_captioning
# Spectral Normalization https://arxiv.org/abs/1802.05957


## Test models
# class RandomAgent(object):
#     def __init__(self, env):
#         self.env = env
#     def step(self, obs):
#         obs = gym.spaces.flatten(self.env.observation_space, obs)
#         print("agent: observation {} shape {} dtype {}\n{}".format(type(obs), obs.shape, obs.dtype, obs))
#         return env.action_space.sample()




## Dreamer
class WorldModel(tf.keras.layers.Layer):
    def __init__(self, env):
        super(WorldModel, self).__init__()

    def call(self, inputs, training=False): # inference/predict
        outputs = {}
        return outputs

class ActionNet(tf.keras.layers.Layer):
    def __init__(self, env):
        super(ActionNet, self).__init__()
        # for space in env.action_space:
        #     pass

        # discrete_classes_count = env.action_space['001_pair'].n

        # event_shape = (1) # sampled output
        # num_components = 64
        # params_size = tfp.layers.MixtureLogistic.params_size(num_components, event_shape)
        # tfp.layers.MixtureLogistic(num_components, event_shape),

        # params_size = tfp.layers.MixtureLogistic.params_size(1, (1))
        # test = tfp.layers.MixtureLogistic(1, (1))

        self.layer_action_dense_in = tf.keras.layers.Dense(128, kernel_initializer='identity', activation='relu', name='action_dense_in')
        self.layer_action_dense_01 = tf.keras.layers.Dense(64, activation='relu', name='action_dense_01')
        self.layer_action_dense_logits_out = tf.keras.layers.Dense(env.action_space['001_pair'].n, kernel_initializer='identity', activation='linear', name='action_dense_logits_out')
        self.layer_action_dist_out = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Categorical(logits=input), name='action_dist_out')

    def call(self, inputs, training=False): # inference/predict
        outputs = {}
        out = self.layer_action_dense_in(inputs['obs'])
        if training: outputs['action_logits'] = out
        outputs['action_dist'] = self.layer_action_dist_out(out)
        return outputs

    # _loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07)
    def loss(self, predicted_value): # maximize value output
        # return self._loss_fn(tf.constant([[2]]), predicted_value)
        return -tf.reduce_sum(predicted_value)

class ValueNet(tf.keras.layers.Layer):
    def __init__(self, env):
        super(ValueNet, self).__init__()
        self.layer_value_dense_in = tf.keras.layers.Dense(128, kernel_initializer='identity', activation='relu', name='value_dense_in')
        self.layer_value_dense_01 = tf.keras.layers.Dense(64, activation='relu', name='value_dense_01')
        self.layer_value_dense_logits_out = tf.keras.layers.Dense(1, kernel_initializer='identity', activation='linear', name='value_dense_logits_out')

    def call(self, inputs, training=False): # inference/predict
        outputs = {}
        out = self.layer_value_dense_in(inputs['obs'])
        outputs['value_logits'] = self.layer_value_dense_logits_out(out)
        return outputs

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07)
    def loss(self, rewards, dones): # top level purpose, for instance "maximize total rewards in the shortest time possible"
        return -tf.reduce_sum(rewards)

class DreamerAI(tf.keras.Model):
    def __init__(self, env):
        super(DreamerAI, self).__init__()
        self.action = ActionNet(env)
        self.value = ValueNet(env)

        self(env_obs_sample(env)) # force the ai_model to build

    @tf.function
    def call(self, inputs, training=False): # inference/predict
        outputs = {}
        outputs = self.action(inputs, training)
        outputs.update(self.value(inputs, training))
        return outputs

    @tf.function
    def step(self, inputs):
        # TODO can imagination run completely in Autograph/GPU? (autograph looping)
        # https://www.tensorflow.org/tutorials/customization/performance#autograph_transformations
        # how do I run imagination constantly and interrupt when new data comes in from the environment?
        with tf.GradientTape() as tape_action, tf.GradientTape() as tape_value:
            outputs = self(inputs, training=True)
            loss_action = self.action.loss(outputs['value_logits'])
            loss_value = self.value.loss(inputs['reward'], inputs['done'])

        gradients_action = tape_action.gradient(loss_action, self.action.trainable_variables)
        gradients_value = tape_value.gradient(loss_value, self.value.trainable_variables)
        self.action.optimizer.apply_gradients(zip(gradients_action, self.action.trainable_variables))
        self.value.optimizer.apply_gradients(zip(gradients_value, self.value.trainable_variables))

        return outputs, loss_action







def env_obs_sample(env):
    rtn = {}
    sample = gym.spaces.flatten(env.observation_space, env.observation_space.sample())
    rtn['obs'] = gpu_tensor_float32(sample)
    # rtn['obs_shape'] = tf.TensorShape([None] + list(sample.shape))
    rtn['obs_shape'] = [None] + list(sample.shape)
    return rtn
# # TODO convert any gym space to tensorflow tensors
# # data = {'six': gym.spaces.Discrete(6), 'bin': gym.spaces.MultiBinary(6)} obs
# def env_space_to_tensor(env, space):
#     pass

class ActorCriticAI(tf.keras.Model):
    def __init__(self, env):
        super(ActorCriticAI, self).__init__()

        self.layer_action_dense_in = tf.keras.layers.Dense(128, kernel_initializer='identity', activation='relu', name='action_dense_in')
        self.layer_action_dense_01 = tf.keras.layers.Dense(64, activation='relu', name='action_dense_01')
        self.layer_action_dense_logits_out = tf.keras.layers.Dense(env.action_space['001_pair'].n, kernel_initializer='identity', activation='linear', name='action_dense_logits_out')
        # self.layer_action_dense_logits_out = tf.keras.layers.Dense(env.action_space.n, kernel_initializer='identity', activation='linear', name='action_dense_logits_out')
        self.layer_action_dist_out = tfp.layers.DistributionLambda(lambda input: tfp.distributions.Categorical(logits=input), name='action_dist_out')

        self.layer_value_dense_in = tf.keras.layers.Dense(128, kernel_initializer='identity', activation='relu', name='value_dense_in')
        self.layer_value_dense_01 = tf.keras.layers.Dense(64, activation='relu', name='value_dense_01')
        self.layer_value_dense_logits_out = tf.keras.layers.Dense(1, kernel_initializer='identity', activation='linear', name='value_dense_logits_out')

        self._obs_sample = env_obs_sample(env)
        self(self._obs_sample) # force the ai_model to build

        self._UPDATE_FREQ = tf.constant(8)
        self._steps = tf.Variable(0)
        self._memory = {}
        self._memory_reset()
    
    def _memory_reset(self):
        self._memory['states'] = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=self._obs_sample['obs_shape'])
        self._memory['actions'] = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=(None,1))
        self._memory['rewards'] = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=(None,1))

    @tf.function
    def call(self, inputs, training=False): # inference/predict
        outputs = {}

        action = self.layer_action_dense_in(inputs['obs'])
        action = self.layer_action_dense_logits_out(action)
        if training: outputs['action_logits'] = action
        outputs['action_dist'] = self.layer_action_dist_out(action)

        value = self.layer_value_dense_in(inputs['obs'])
        outputs['value_logits'] = self.layer_value_dense_logits_out(value)

        return outputs

    def compute_loss(self, inputs, outputs, gamma=0.99):
        return 0.0

        # if done: reward_sum = 0.0  # terminal
        # else: reward_sum = self(inputs)




        # if done: reward_sum = 0.0  # terminal
        # else: reward_sum = self.local_model(
        #     tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]

        # # Get discounted rewards
        # discounted_rewards = []
        # for reward in memory.rewards[::-1]:  # reverse buffer r
        # reward_sum = reward + gamma * reward_sum
        # discounted_rewards.append(reward_sum)
        # discounted_rewards.reverse()

        # logits, values = self.local_model(
        #     tf.convert_to_tensor(np.vstack(memory.states),
        #                         dtype=tf.float32))
        # # Get our advantages
        # advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
        #                         dtype=tf.float32) - values
        # # Value loss
        # value_loss = advantage ** 2

        # # Calculate our policy loss
        # actions_one_hot = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)

        # policy = tf.nn.softmax(logits)
        # entropy = tf.reduce_sum(policy * tf.log(policy + 1e-20), axis=1)

        # policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot,
        #                                                         logits=logits)
        # policy_loss *= tf.stop_gradient(advantage)
        # policy_loss -= 0.01 * entropy
        # total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        # return total_loss

    @tf.function
    def step(self, inputs):
        # TODO fix these bools in here to work with graph
        if inputs['action_old'] is not None:
            self._memory['states'].write(self._memory['states'].size(), inputs['obs_old'])
            self._memory['actions'].write(self._memory['actions'].size(), inputs['action_old'])
            self._memory['rewards'].write(self._memory['rewards'].size(), inputs['reward'])

        test = (inputs['done'][0][0] is True) or (self._steps % self._UPDATE_FREQ == 0)
        if inputs['action_old'] is None or not test:
            print('inference scan path')
            tf.print('inference')
            outputs = self(inputs) # no train, just get action, value
        else:
            print('train scan path')
            tf.print('train')
            # with tf.GradientTape() as tape:
            #     outputs = self(inputs, training=True)
            #     loss = self.compute_loss(inputs, outputs)
            # gradients = tape.gradient(loss, self.trainable_variables)
            # self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            outputs = self(inputs, training=True)
            outputs['states'] = self._memory['states'].concat()
            outputs['actions'] = self._memory['actions'].concat()
            outputs['rewards'] = self._memory['rewards'].concat()

            self._memory_reset()
            # outputs['loss'] = loss

        self._steps.assign_add(1)
        return outputs


# convert numpy array to tf.float32 tensor and put on GPU
# TODO make these into one function?
@tf.function
def gpu_tensor_float32(a):
    return tf.expand_dims(tf.convert_to_tensor(a, dtype=tf.float32), 0)
@tf.function
def gpu_tensor_int32(a):
    return tf.expand_dims(tf.convert_to_tensor(a, dtype=tf.int32), 0)
@tf.function
def gpu_tensor_bool(a):
    return tf.expand_dims(tf.convert_to_tensor(a, dtype=tf.bool), 0)

# TODO convert to tf
def _timestamp_convert_for_input(self, t): # unix (epoc) UTC timestamp float with microseconds (time.time())
    f, i = np.modf(np.float64(t) / 188) # 100 year range from 1970 376
    f = f * 2 - 1 # -1.0:1.0 range
    i = i / 2**23 - 1 # fit high value into 23 mantissa + 1 sign bits (float32) at -1.0:1.0 range
    return tf.float32([f,i]) # [low(float32), high(float32)]
def _timestamp_convert_from_output(self, low, high): # [low(float32), high(float32)]
    # TODO
    return np.float64(0.0) # unix (epoc) timestamp float with microseconds (time.time())

## Generic agent that uses AI models
class AIAgent(object):
    def __init__(self, model, env):
        self.model, self.env = model, env
        self.obs_old = env_obs_sample(env)['obs']
        self.action_old = gpu_tensor_int32([env.action_space.sample()])
    def step(self, obs, reward, done, info): # all new info
        # TODO this is where native data goes into GPU
        # TODO loop through and send in seperate items in observation
        obs = gym.spaces.flatten(self.env.observation_space, obs)
        # print("agent: observation {} shape {} dtype {}\n{}".format(type(obs), obs.shape, obs.dtype, obs))

        obs = gpu_tensor_float32(obs)

        input_buffer = {}
        input_buffer['obs_old'] = self.obs_old
        input_buffer['action_old'] = self.action_old
        input_buffer['done'] = gpu_tensor_bool([done])
        input_buffer['reward'] = gpu_tensor_float32([reward])
        input_buffer['obs'] = obs

        outputs = self.model.step(input_buffer)
        if 'rewards' in outputs: # print(outputs['rewards'].shape)
            test = outputs['rewards']
            print("test type {} shape {} dtype {} device {}\n{}\n\n".format(type(test), test.shape, test.dtype, test.device, test))

        self.obs_old = obs
        self.action_old = gpu_tensor_int32(outputs['action_dist'])

        action = env.action_space.sample()
        action['001_pair'] = int(outputs['action_dist'][0])
        return action





## Main loop
if __name__ == '__main__':
    me = 0
    # env = gym.make('FrozenLake-v0') # Discrete(16)	Discrete(4)	(0, 1)	100	100	0.78
    # env, model_name = gym.make('CartPole-v0'), "gym-ActorCritic-CartPole" # Box(4,)	Discrete(2)	(-inf, inf)	200	100	195.0
    # env, model_name = gym.make('LunarLander-v2'), "gym-ActorCritic-LunarLander" # Box(8,)	Discrete(4)	(-inf, inf)	1000	100	200
    # env = gym.make('LunarLanderContinuous-v2') # Box(8,)	Box(2,)	(-inf, inf)	1000	100	200
    # env = gym.make('CarRacing-v0') # Box(96, 96, 3)	Box(3,)	(-inf, inf)	1000	100	900
    # env = gym.make('MontezumaRevengeNoFrameskip-v4') # Box(210, 160, 3)	Discrete(18)	(-inf, inf)	400000	100	None
    env, model_name = gym.make('Trader-v0'), "gym-ActorCritic-Trader"
    env = gym.make('Trader-v0', agent_id=me)
    env.seed(0)

    # model = DreamerAI(env)
    model = ActorCriticAI(env)
    
    model_file = "{}/tf_models/{}-{}.h5".format(curdir, model_name, me)
    if tf.io.gfile.exists(model_file):
        model.load_weights(model_file, by_name=True, skip_mismatch=True)
        print("LOADED model weights from {}".format(model_file))

    agent = AIAgent(model, env)
    reward, done, info = 0.0, True, {}
    for i_episode in range(3):
        # TODO env.reset and env.step could take lots of time, and ai_model needs to run independently, need to figure out threading/multiprocessing
        obs = env.reset()
        reward_episode = 0.0
        # print("{}\n".format(obs))
        # env.render()

        for t_timesteps in range(100):
            action = agent.step(obs, reward, done, info)

            obs, reward, done, info = env.step(action)
            reward_episode += reward
            # print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))
            env.render()

            # time.sleep(1.01)
            if done: break
        print("agent: {}episode {} | timesteps {} | reward mean {} total {}\n".format(('DONE ' if done else ''), i_episode+1, t_timesteps+1, reward_episode/(t_timesteps+1), reward_episode))

    agent.step(obs, reward, done, info) # learn from the last episode
    env.close()

    model.summary()
    model.save_weights(model_file)
