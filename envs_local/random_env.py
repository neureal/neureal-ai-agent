import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gym
import gym_util

# TODO auto make the dtype struc from space for numpy dtype compatability with gym, need to include space it has more info like low,high
# def gym_space_to_dtype(space):
#     pass

class RandomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env_np_struc):
        super(RandomEnv, self).__init__()
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.reward_range = (-np.inf,+np.inf)
        # self.obs_zero = gym.spaces.flatten(self.observation_space, self.observation_space.sample())
        # self.action_spec, self.action_zero, self.action_zero_out = gym_util.get_spec(self.action_space)
        # self.obs_spec, self.obs_zero, self.obs_zero_out = gym_util.get_spec(self.observation_space)
        # obs_dtype = gym_space_to_dtype(self.observation_space)
        # action_smpl = self.action_space.sample()
        # obs_smpl = self.observation_space.sample()

        if env_np_struc: self.np_struc = True
        if env_np_struc:
            action_dtype = self._action_space_struc()
            obs_dtype = self._obs_space_struc()
            self.action_dtype, self.obs_dtype = action_dtype, obs_dtype
            action_zero = np.zeros((1,), self.action_dtype)
            obs_zero = np.zeros((1,), self.obs_dtype)
        else:
            action_zero = gym_util.get_space_zero(self.action_space)
            obs_zero = gym_util.get_space_zero(self.observation_space)
        self.action_zero, self.obs_zero = action_zero, obs_zero

        self.state = self.action_zero, self.obs_zero, np.float64(0.0), False, {}

    def step(self, action):
        return self._request(action)
    def reset(self):
        return self._request(None)[0]
    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.state
        if action is None: print("{}\n".format(obs))
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

        # action_space = gym.spaces.Tuple([])

        # action_space.spaces.append(gym.spaces.Discrete(4))

        # action_space_sub = gym.spaces.Tuple([])
        # action_space_sub.spaces.append(gym.spaces.Discrete(8))
        # action_space_sub.spaces.append(gym.spaces.Box(low=0, high=255, shape=(3,2), dtype=np.uint8))
        # action_space.spaces.append(action_space_sub)

        # action_space.spaces.append(gym.spaces.Discrete(6))

        # action_space_sub2 = gym.spaces.Dict()
        # action_space_sub2.spaces['test'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float64)
        # action_space.spaces.append(action_space_sub2)

        action_space = gym.spaces.Dict()

        action_space.spaces['dt_sub'] = gym.spaces.Dict()
        action_space.spaces['dt_sub'].spaces['float64'] = gym.spaces.Box(low=np.NINF, high=np.inf, shape=(2,), dtype=np.float64)
        action_space.spaces['dt_sub'].spaces['byte'] = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)

        action_space.spaces['byte'] = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        action_space.spaces['2Darray'] = gym.spaces.Box(low=0, high=255, shape=(2,3), dtype=np.uint8)
        action_space.spaces['discrete6'] = gym.spaces.Discrete(6)
        action_space.spaces['float64'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
        action_space.spaces['bools'] = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.bool)
        

        return action_space

    def _action_space_struc(self):
        dt_sub = np.dtype([
            ('float64', np.float64, (2,)),
            ('byte', np.uint8, (2,)),
        ])
        dtype = np.dtype([
            ('dt_sub', dt_sub),
            ('byte', np.uint8),
            ('2Darray', np.uint8, (2,3)),
            ('discrete6', np.int64),
            ('float64', np.float64),
            ('bools', np.bool, (5,)),
        ])
        return dtype



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

        # obs_space = gym.spaces.Dict()
        # obs_space.spaces['timestamp'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
        # obs_space.spaces['data'] = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)
        
        # obs_space_sub = gym.spaces.Tuple([])
        # obs_space_sub.spaces.append(gym.spaces.Discrete(8))
        # obs_space_sub.spaces.append(gym.spaces.Box(low=0, high=255, shape=(3,2), dtype=np.uint8))
        # obs_space.spaces['extra'] = obs_space_sub

        obs_space = gym.spaces.Dict()
        # obs_space.spaces['timestamp'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
        obs_space.spaces['byte'] = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        obs_space.spaces['2Darray'] = gym.spaces.Box(low=0, high=255, shape=(2,3), dtype=np.uint8)
        obs_space.spaces['float32'] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_space.spaces['discrete6'] = gym.spaces.Discrete(6)
        obs_space.spaces['float64'] = gym.spaces.Box(low=np.NINF, high=np.inf, shape=(1,), dtype=np.float64)
        obs_space.spaces['bools'] = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.bool)
        obs_space.spaces['image'] = gym.spaces.Box(low=0, high=255, shape=(3,3,3), dtype=np.uint8)
        for i in range(2):
            obs_space.spaces['v'+str(i)] = gym.spaces.Dict()
            obs_space.spaces['v'+str(i)].spaces['float64'] = gym.spaces.Box(low=np.NINF, high=np.inf, shape=(1,), dtype=np.float64)
            obs_space.spaces['v'+str(i)].spaces['byte'] = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            obs_space.spaces['t'+str(i)] = gym.spaces.Dict()
            obs_space.spaces['t'+str(i)].spaces['float16'] = gym.spaces.Box(low=np.NINF, high=np.inf, shape=(3,1), dtype=np.float16)
            obs_space.spaces['t'+str(i)].spaces['byte'] = gym.spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)

        return obs_space

    def _obs_space_struc(self):
        types = [
            # ('timestamp', np.float64),
            ('byte', np.uint8),
            ('2Darray', np.uint8, (2,3)),
            ('float32', np.float32),
            ('discrete6', np.int64),
            ('float64', np.float64, (1,)),
            ('bools', np.bool, (5,)),
            ('image', np.uint8, (3,3,3)),
        ]
        dt_sub = np.dtype([
            ('float64', np.float64, (1,)),
            ('byte', np.uint8),
        ])
        dt_itm = np.dtype([
            ('float16', np.float16, (3,1)),
            ('byte', np.uint8, (3,)),
        ])
        for i in range(2):
            types.append(('v'+str(i), dt_sub))
            types.append(('t'+str(i), dt_itm))
        dtype = np.dtype(types)
        return dtype



    def _request(self, action):
        obs = self.obs_zero
        reward = np.float64(0.0)
        done = False
        info = {}
        
        # if action is None: print("RandomEnv reset")
        
        if hasattr(self,'np_struc'):
            obs = np.random.randint(32, size=self.obs_dtype.itemsize, dtype=np.uint8)
            obs = np.frombuffer(obs, dtype=self.obs_dtype)
            # obs = np.zeros((1,), self.obs_dtype)
            # # obs = np.where(np.isnan(obs), 0, obs)
        else:
            obs = self.observation_space.sample()
            # obs = gym.spaces.flatten(self.observation_space, self.observation_space.sample())
            # obs = {
            #     'matrix': np.random.pareto(1.0, size=(4,4)),
            #     'text': np.random.randint(low=0, high=256, size=(16,), dtype=np.uint8),
            #     'features': np.random.normal(0.0, 1.0, size=(4,)),
            # }
            # obs = np.zeros(shape=self.observation_space.shape, dtype=self.observation_space.dtype)
        reward = np.float64(np.random.standard_normal())
        # reward = np.float64(np.random.standard_cauchy())
        if np.random.randint(10) >= 9: done = True

        self.state = (action, obs, reward, done, info)
        return obs, reward, done, info



if __name__ == '__main__':
    ## test
    env_np_struc = True
    env = RandomEnv(env_np_struc)
    obs = env.reset()
    # env.render()
    if hasattr(env,'np_struc'):
        # test_spec, test_obs_zero, test_obs_zero_out = gym_util.get_spec(env.observation_space, compute_dtype='float32')
        # test_obs = gym_struc_to_feat(obs)

        action = np.random.randint(32, size=env.action_dtype.itemsize, dtype=np.uint8)
        action = np.frombuffer(action, dtype=env.action_dtype)
        # action = np.zeros((2,), env.action_dtype)
        # print("{}".format(action))
        test_out = gym_util.struc_to_feat(action)
        # print("{}".format(test_out))
        test_action = gym_util.out_to_struc(test_out, env.action_dtype)
        # print("{}".format(test_action))
    else:
        test_spec, test_obs_zero, test_obs_zero_out = gym_util.get_spec(env.observation_space, compute_dtype='float32')
        test_obs = gym_util.space_to_feat(obs, env.observation_space)

        action = env.action_space.sample()
        # test_out = [np.asarray([2],np.int64), np.asarray([3],np.int64)]
        test_out = gym_util.space_to_feat(action, env.action_space)
        test_action = gym_util.out_to_space(test_out, env.action_space, [0])

    obs, reward, done, info = env.step(action)
    # env.render()
