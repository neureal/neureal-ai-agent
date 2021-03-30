import time
import multiprocessing as mp
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gym


class AsyncWrapperEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env):
        super(AsyncWrapperEnv, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

        # TODO setup process so agent can start using reset()
        # self.proc = mp.Process(target=self._proc_run, name='ENV_PROC', args=(lock_print, process_ctrl, action, obs, reward))
        # process_ctrl.value = 0
        # self.proc.join()

    def _proc_run(self):

    def seed(self): return self.env.seed()
    def close(self): return self.env.close()
    def render(self, mode='human', close=False): return self.env.render(mode, close)
    def reset(self):
        # TODO start new process and step env with env.action_noop (stop old process loop)
        # self.proc.start()
        return self.env.reset()
    def step(self, action):
        # TODO insert this action into noop stream
        return self.env.step(action)
