import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gym


class RandomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(RandomEnv, self).__init__()
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.observation_space_zero = self.observation_space.sample()
        # for space in self.observation_space_zero: self.observation_space_zero[space].fill(0)
        # self.observation_space_zero = gym.spaces.flatten(self.observation_space, self.observation_space.sample())
        self.observation_space_zero.fill(0)
        # self.reward_range = (-inf,+inf)
    def step(self, action):
        return self._request(action)
    def reset(self):
        return self._request(None)[0]
    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.state
        if action == None: print("{}\n".format(obs))
        else: print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))

    def _action_space(self):
        # return gym.spaces.Dict({ # sorted by name
        #     '001_six': gym.spaces.Discrete(6), # int
        #     '002_bin': gym.spaces.MultiBinary(6), # np.ndarray([1, 0, 1, 0, 0, 1], dtype=int8)
        #     '003_mul': gym.spaces.MultiDiscrete([6,2]), # np.ndarray([3, 0], dtype=int64)
        #     '004_val': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float64)} # np.ndarray(2.22744361)
        # )
        return gym.spaces.Discrete(6)
        # return gym.spaces.MultiDiscrete([6,2])
        # return gym.spaces.MultiBinary(6)
        # return gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.MultiBinary(6)))
        # return gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)
        # return gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    def _observation_space(self):
        # return gym.spaces.Dict({ # sorted by name
        #     # '001_minst': gym.spaces.Box(low=0, high=255, shape=(28,28), dtype=np.uint8),
        #     '002_matrix': gym.spaces.Box(low=0.0, high=np.inf, shape=(4,4), dtype=np.float64),
        #     '003_text': gym.spaces.Box(low=0, high=255, shape=(16,), dtype=np.uint8),
        #     '004_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
        # })
        # return gym.spaces.Dict({'six': gym.spaces.Discrete(6), 'bin': gym.spaces.MultiBinary(6)})
        # return gym.spaces.Box(low=0, high=255, shape=(8,), dtype=np.uint8)
        # return gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        # return gym.spaces.Box(low=0.0, high=np.inf, shape=(4,), dtype=np.float64)
    def _request(self, action):
        # obs = np.zeros(shape=self.observation_space.shape, dtype=self.observation_space.dtype)
        obs = self.observation_space_zero
        reward = np.float64(0.0)
        done = False
        info = {}

        obs = self.observation_space.sample()
        # obs = gym.spaces.flatten(self.observation_space, self.observation_space.sample())
        # obs = {
        #     '002_matrix': np.random.pareto(1.0, size=(4,4)),
        #     '003_text': np.random.randint(low=0, high=256, size=(16,), dtype=np.uint8),
        #     '004_features': np.random.normal(0.0, 1.0, size=(4,)),
        # }
        reward = np.float64(np.random.standard_normal())

        self.state = (action, obs, reward, done, info)
        return obs, reward, done, info