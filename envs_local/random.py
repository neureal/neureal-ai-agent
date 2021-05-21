import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gym
import model_util as util


class RandomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(RandomEnv, self).__init__()
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.reward_range = (-np.inf,+np.inf)

        obs_smpl = self.observation_space.sample()
        action_smpl = self.action_space.sample()

        obs_zero = util.gym_get_space_zero(self.observation_space)
        action_zero = util.gym_get_space_zero(self.action_space)
        self.action_zero, self.obs_zero = action_zero, obs_zero
        # self.obs_spec, self.obs_zero, self.obs_zero_out = util.gym_get_spec(self.observation_space)
        # self.action_spec, self.action_zero, self.action_zero_out = util.gym_get_spec(self.action_space)
        self.state = self.action_zero, self.obs_zero, np.float64(0.0), False, {}
        # self.obs_zero = gym.spaces.flatten(self.observation_space, self.observation_space.sample())
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
        #     'six': gym.spaces.Discrete(6), # int
        #     'bin': gym.spaces.MultiBinary(6), # np.ndarray([1, 0, 1, 0, 0, 1], dtype=int8)
        #     'mul': gym.spaces.MultiDiscrete([6,2]), # np.ndarray([3, 0], dtype=int64)
        #     'val': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float64)} # np.ndarray(2.22744361)
        # )
        # return gym.spaces.Discrete(6)
        # return gym.spaces.MultiDiscrete([6,2])
        # return gym.spaces.MultiBinary(6)
        # return gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.MultiBinary(6)))
        # return gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)
        # return gym.spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float64)

        action_space = gym.spaces.Tuple([])

        action_space.spaces.append(gym.spaces.Discrete(4))

        action_space_sub = gym.spaces.Tuple([])
        action_space_sub.spaces.append(gym.spaces.Discrete(8))
        action_space_sub.spaces.append(gym.spaces.Box(low=0, high=255, shape=(3,2), dtype=np.uint8))
        action_space.spaces.append(action_space_sub)

        action_space.spaces.append(gym.spaces.Discrete(6))

        action_space_sub2 = gym.spaces.Dict()
        action_space_sub2.spaces['test'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float64)
        action_space.spaces.append(action_space_sub2)

        return action_space

    def _observation_space(self):
        # return gym.spaces.Dict({ # sorted by name
        #     # 'minst': gym.spaces.Box(low=0, high=255, shape=(28,28), dtype=np.uint8),
        #     'matrix': gym.spaces.Box(low=0.0, high=np.inf, shape=(4,4), dtype=np.float64),
        #     'text': gym.spaces.Box(low=0, high=255, shape=(16,), dtype=np.uint8),
        #     'features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
        # })
        # return gym.spaces.Dict({'six': gym.spaces.Discrete(6), 'bin': gym.spaces.MultiBinary(6)})
        # return gym.spaces.Box(low=0, high=255, shape=(8,), dtype=np.uint8)
        # return gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        # return gym.spaces.Box(low=0.0, high=np.inf, shape=(4,), dtype=np.float64)
        # return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        obs_space = gym.spaces.Dict()
        obs_space.spaces['timestamp'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
        obs_space.spaces['data'] = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)
        
        obs_space_sub = gym.spaces.Tuple([])
        obs_space_sub.spaces.append(gym.spaces.Discrete(8))
        obs_space_sub.spaces.append(gym.spaces.Box(low=0, high=255, shape=(3,2), dtype=np.uint8))
        obs_space.spaces['extra'] = obs_space_sub

        return obs_space

    def _request(self, action):
        # obs = np.zeros(shape=self.observation_space.shape, dtype=self.observation_space.dtype)
        obs = self.obs_zero
        reward = np.float64(0.0)
        done = False
        info = {}

        obs = self.observation_space.sample()
        # obs = gym.spaces.flatten(self.observation_space, self.observation_space.sample())
        # obs = {
        #     'matrix': np.random.pareto(1.0, size=(4,4)),
        #     'text': np.random.randint(low=0, high=256, size=(16,), dtype=np.uint8),
        #     'features': np.random.normal(0.0, 1.0, size=(4,)),
        # }
        reward = np.float64(np.random.standard_normal())
        # reward = np.float64(np.random.standard_cauchy())
        if np.random.randint(10) >= 9: done = True

        self.state = (action, obs, reward, done, info)
        return obs, reward, done, info
        
if __name__ == '__main__':
    ## test
    env = RandomEnv()
    obs = env.reset()
    env.render()
    action = env.action_space.sample()
    # test_out = [np.asarray([2],np.int64), np.asarray([3],np.int64)]
    # test_out = util.gym_obs_to_feat(action)
    # test_action = util.gym_out_to_action(test_out, env.action_space)
    obs, reward, done, info = env.step(action)
    test_spec, test_obs_zero, test_obs_zero_out = util.gym_get_spec(env.observation_space, compute_dtype='float32')
    test = util.gym_obs_to_feat(obs, env.observation_space)
    env.render()
