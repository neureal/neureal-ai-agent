# from collections import OrderedDict
# import copy
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gym, gym_algorithmic, procgen, pybullet_envs
# import gym_util


class ReconfigWrapperEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env):
        super(ReconfigWrapperEnv, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

        self.reconfig = False
        if isinstance(env.observation_space, gym.spaces.Box):
            self.reconfig = True
            num_feat = env.observation_space.shape[-1]; self.num_feat = num_feat
            obs_shape = env.observation_space.shape[:-1] + (1,)
            obs = gym.spaces.Tuple([])
            for i in range(num_feat):
                feat = gym.spaces.Box(
                    low=env.observation_space.low[...,i:i+1],
                    high=env.observation_space.high[...,i:i+1],
                    shape=obs_shape,
                    dtype=env.observation_space.dtype
                )
                obs.spaces.append(feat)
            obs.spaces.append(env.observation_space)
            self.observation_space = obs


    def seed(self): return self.env.seed()
    def render(self, mode='human', close=False): return self.env.render(mode, close)
    def close(self): return self.env.close()

    def reset(self):
        obs = self.env.reset()
        if self.reconfig:
            obs_ = [None]*(self.num_feat+1)
            for i in range(self.num_feat): obs_[i] = obs[...,i:i+1]
            obs_[-1] = obs
            obs = tuple(obs_)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.reconfig:
            obs_ = [None]*(self.num_feat+1)
            for i in range(self.num_feat): obs_[i] = obs[...,i:i+1]
            obs_[-1] = obs
            obs = tuple(obs_)
        return obs, reward, done, info

if __name__ == '__main__':
    ## test
    env = gym.make('CartPole-v0') # ; env.observation_space.dtype = np.dtype('float64')
    # env = gym.make('procgen-chaser-v0')
    env = ReconfigWrapperEnv(env)
    obs = env.reset()
    # print("main reset", obs)
    action = env.action_space.sample()
    # print("main action", action)
    obs, reward, done, info = env.step(action)
    # print("main step ", obs)
    env.close()
    print("done")
