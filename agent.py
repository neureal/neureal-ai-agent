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

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

# include print_time, fixinfnan, EvoNormS0 from "wilutil" library

class ActorCriticAI(tf.keras.Model):
    def __init__(self, env):
        super(ActorCriticAI, self).__init__()
    @tf.function
    def call(self, inputs, training=None):
        pass


class Agent:
    def __init__(self, model, env):
        self.model, self.env = model, env
    def vivify(self):
        # (train_step)
        pass

# class AgentDreamer:

if __name__ == '__main__':
    ## manage multiprocessing
    # load models
    with tf.device("CPU:0"):
        model = ActorCriticAI(env)
        model_name += "-{}-{}-a{}".format(model.net_arch, machine, device)
        model_file = "{}/tf-data-models-local/{}.h5".format(curdir, model_name);
        if tf.io.gfile.exists(model_file): model.load_weights(model_file, by_name=True, skip_mismatch=True)
    # setup ctrl,data,param sharing
    # start agents (real+dreamers)
    agent = Agent(model)
    # agent_process = mp.Process(target=agent.vivify, name='AGENT', args=(lock_print, process_ctrl, weights_shared))
    # agent_process.start()
    # quit on keyboard (space = save, esc = no save)
    process_ctrl.value = 0
    agent_process.join()
    # plot results
    # save models
    model.save_weights(model_file)
