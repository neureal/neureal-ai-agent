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
    def seed(self):
        return self.env.seed()
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def render(self, mode='human', close=False):
        return self.env.render(mode, close)
    def close(self):
        return self.env.close()
