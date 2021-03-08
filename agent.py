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
import gym, gym_trader

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

# TODO use numba to make things faster on CPU

class ActionNet(tf.keras.layers.Layer):
    def __init__(self, env):
        super(ActionNet, self).__init__()
        self.layer_dense_in = tf.keras.layers.Dense(1024, activation=util.EvoNormS0(16), use_bias=False, name='dense_in')
        self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
    @tf.function
    def call(self, inputs, training=None):
        out = tf.cast(inputs['obs'], dtype=self.compute_dtype)
        out = self.layer_dense_in(out)
        out = self.layer_dense_out(out)
        return out
        
class ValueNet(tf.keras.layers.Layer):
    def __init__(self, env):
        super(ValueNet, self).__init__()
        self.layer_dense_in = tf.keras.layers.Dense(1024, activation=util.EvoNormS0(16), use_bias=False, name='dense_in')
        self.layer_dense_out = tf.keras.layers.Dense(1, name='dense_out')
    @tf.function
    def call(self, inputs, training=None):
        out = tf.cast(inputs['obs'], dtype=self.compute_dtype)
        out = self.layer_dense_in(out)
        out = self.layer_dense_out(out)
        return out

class GeneralAI(tf.keras.Model):
    def __init__(self, env, learn_rate):
        super(GeneralAI, self).__init__()
        
        inputs = {} # must pre build to create variables outside @tf.function
        inputs['obs'], inputs['reward'], inputs['done'] = tf.zeros([1]+list(env.observation_space.shape), dtype=self.obs_dtype), tf.constant([[0]], dtype=tf.float64), tf.constant([[False]], dtype=tf.bool)
        self.action = ActionNet(env); outputs = self.action(inputs)
        self.value = ValueNet(env); outputs = self.value(inputs)

    @tf.function
    def A2C_infer(self, inputs, training=None):
        print("tracing -> GeneralAI A2C_infer")


env_name = 'CartPole-v0' # Box((4),-inf:inf,float32)         Discrete(2,int64)
num_agents = 1
learn_rate = 1e-4

if __name__ == '__main__':
    ## manage multiprocessing
    with tf.device('/device:GPU:0'):
        env = gym.make(env_name)
        model = GeneralAI(env, learn_rate=learn_rate)

        # load models, load each net seperately
        # model_name += "-{}-{}-a{}".format(model.net_arch, machine, device)

        # model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name); loaded_model = False
        # if tf.io.gfile.exists(model_file):
        #     model.load_weights(model_file, by_name=True, skip_mismatch=True)
        #     print("LOADED model weights from {}".format(model_file)); loaded_model = True
        # # print(model.call.pretty_printed_concrete_signatures()); quit(0)
        # # model.summary(); quit(0)


        # # setup ctrl,data,param sharing
        # # start agents (real+dreamers)
        # agent = Agent(model)
        # # agent_process = mp.Process(target=agent.vivify, name='AGENT', args=(lock_print, process_ctrl, weights_shared))
        # # agent_process.start()
        # # quit on keyboard (space = save, esc = no save)
        # process_ctrl.value = 0
        # agent_process.join()
        # # plot results
        # # save models
        # model.save_weights(model_file)
