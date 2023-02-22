from collections import OrderedDict
import time
import multiprocessing as mp
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gymnasium as gym
import gym_util


class AsyncWrapperEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env, env_clock, env_speed, env_render):
        super(AsyncWrapperEnv, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

        np_struc = hasattr(env,'np_struc')
        if np_struc: self.np_struc, self.action_dtype, self.obs_dtype = env.np_struc, env.action_dtype, env.obs_dtype

        self._env_clock, self._env_speed, self._env_render, self._env_np_struc = env_clock, env_speed, env_render, np_struc
        reward_done_zero = [np.frombuffer(np.asarray(0, np.float64), dtype=np.uint8), np.frombuffer(np.asarray(False, bool), dtype=np.uint8)]
        self._reward_done_zero = reward_done_zero

        self._action_timing, self._obs_timing = False, False
        if not (isinstance(env.action_space, gym.spaces.Dict) and 'timedelta' in env.action_space.spaces):
            self._action_timing = True
            self.action_space = gym.spaces.Dict()
            self.action_space.spaces['timedelta'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
            self.action_space.spaces['origspace'] = env.action_space
            if np_struc: self.action_dtype = np.dtype([('timedelta', 'f8'), ('origspace', env.action_dtype)])
        if not (isinstance(env.observation_space, gym.spaces.Dict) and 'timestamp' in env.observation_space.spaces):
            self._obs_timing = True
            self.observation_space = gym.spaces.Dict()
            self.observation_space.spaces['timestamp'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
            self.observation_space.spaces['origspace'] = env.observation_space
            if np_struc: self.obs_dtype = np.dtype([('timestamp', 'f8'), ('origspace', env.obs_dtype)])

        if np_struc:
            reward_size, done_size = np.dtype(np.float64).itemsize, np.dtype(bool).itemsize
            self._obs_idx, self._done_idx = -(reward_size + done_size), -done_size
            action_size, obs_size = env.action_dtype.itemsize, self.obs_dtype.itemsize + reward_size + done_size
        else:
            idx = 0; action_idxs = gym_util.space_to_bytes(env.action_space.sample(), env.action_space)
            for i in range(len(action_idxs)): idx += action_idxs[i].size; action_idxs[i] = idx
            action_idxs = [0] + action_idxs

            idx = 0; obs_idxs = gym_util.space_to_bytes(self.observation_space.sample(), self.observation_space)
            obs_idxs += reward_done_zero
            for i in range(len(obs_idxs)): idx += obs_idxs[i].size; obs_idxs[i] = idx
            obs_idxs = [0] + obs_idxs

            self._action_idxs, self._obs_idxs = action_idxs, obs_idxs
            action_size, obs_size = action_idxs[-1], obs_idxs[-1]

        # self._lock_print = mp.Lock()
        self._proc_ctrl = mp.Value('b', 0) # 1 = close, 0 = reset, -1 = step, -2 = done
        self._action_shared = mp.sharedctypes.Array('B', action_size)
        self._obs_shared = mp.sharedctypes.Array('B', obs_size)
        self._proc = mp.Process(target=self._proc_run, name='ENV', args=())


    def _translate_obs(self, obs, reward_done):
        if self._obs_timing:
            timestamp = np.float64(time.time() * self._env_speed)
            if self._env_np_struc:
                obs = np.asarray([(timestamp, obs)], dtype=self.obs_dtype)
            else:
                obs = OrderedDict({'timestamp': timestamp, 'origspace': obs})
        if self._env_np_struc: obs = [np.frombuffer(obs, dtype=np.uint8)]
        else: obs = gym_util.space_to_bytes(obs, self.observation_space)
        obs += reward_done
        obs = np.concatenate(obs)
        return obs

    def _proc_run(self):
        action_view = np.asarray(self._action_shared.get_obj())
        obs_view = np.asarray(self._obs_shared.get_obj())
        action = np.zeros(action_view.shape, action_view.dtype)

        while self._proc_ctrl.value != 1:

            if self._proc_ctrl.value == 0:
                obs, info = self.env.reset()
                # print("proc reset", obs)
                obs = self._translate_obs(obs, self._reward_done_zero)
                with self._obs_shared.get_lock():
                    if not np.array_equal(obs_view, obs): np.copyto(obs_view, obs, casting='no')
                self._proc_ctrl.value = -1

            elif self._proc_ctrl.value == -1:
                with self._action_shared.get_lock():
                    if not np.array_equal(action_view, action): np.copyto(action, action_view, casting='no')
                if self._env_np_struc: action_space = np.frombuffer(action, dtype=self.env.action_dtype)
                else: action_space = gym_util.bytes_to_space(action, self.env.action_space, self._action_idxs, [0])

                # print("proc action", action_space)
                obs, reward, terminated, truncated, _ = self.env.step(action_space); done = (terminated or truncated)
                # print("proc step", obs)
                reward_done = [np.frombuffer(np.asarray(reward, np.float64), dtype=np.uint8), np.frombuffer(np.asarray(done, bool), dtype=np.uint8)]
                obs = self._translate_obs(obs, reward_done)
                with self._obs_shared.get_lock():
                    if not np.array_equal(obs_view, obs): np.copyto(obs_view, obs, casting='no')
                if done: self._proc_ctrl.value = -2

            if self._env_render: self.env.render()
            time.sleep(self._env_clock)

        self.env.close()


    # def seed(self): return self.env.seed()
    # def render(self, mode='human', close=False): return self.env.render(mode, close)
    def seed(self): return
    def render(self, mode='human', close=False): return

    def close(self):
        self._proc_ctrl.value = 1
        self._proc.join()
    def reset(self):
        obs_view = np.asarray(self._obs_shared.get_obj())
        obs = np.zeros(obs_view.shape, obs_view.dtype)

        # if self._proc.is_alive(): self.proc_stop()
        # self._proc = mp.Process(target=self._proc_run, name='ENV', args=())
        self._proc_ctrl.value = 0
        if not self._proc.is_alive(): self._proc.start()
        while self._proc_ctrl.value != -1: time.sleep(0.0)

        with self._obs_shared.get_lock(): np.copyto(obs, obs_view, casting='no')
        if self._env_np_struc: obs = np.frombuffer(obs[:self._obs_idx], dtype=self.obs_dtype)
        else: obs = gym_util.bytes_to_space(obs, self.observation_space, self._obs_idxs, [0])
        return obs, {}

    def step(self, action):
        action_view = np.asarray(self._action_shared.get_obj())
        obs_view = np.asarray(self._obs_shared.get_obj())
        obs = np.zeros(obs_view.shape, obs_view.dtype)

        # TODO try different ways to include timing
        timedelta = 0.0
        if self._action_timing:
            timedelta = action['timedelta'][0] / self._env_speed
            action = action['origspace']
        if self._env_np_struc: action = np.frombuffer(action, dtype=np.uint8)
        else:
            action = gym_util.space_to_bytes(action, self.env.action_space)
            action = np.concatenate(action)
        with self._action_shared.get_lock():
            if not np.array_equal(action_view, action): np.copyto(action_view, action, casting='no')
        # print(timedelta)
        time.sleep(timedelta)

        with self._obs_shared.get_lock(): np.copyto(obs, obs_view, casting='no')
        if self._env_np_struc:
            reward = obs[self._obs_idx:self._done_idx]
            done = obs[self._done_idx:]
            obs = np.frombuffer(obs[:self._obs_idx], dtype=self.obs_dtype)
        else:
            reward = obs[self._obs_idxs[-3]:self._obs_idxs[-2]]
            done = obs[self._obs_idxs[-2]:self._obs_idxs[-1]]
            obs = gym_util.bytes_to_space(obs, self.observation_space, self._obs_idxs, [0])
        reward = np.frombuffer(reward, dtype=np.float64).item()
        done = np.frombuffer(done, dtype=bool).item()
        return obs, reward, done, False, {}

if __name__ == '__main__':
    ## test
    # env = gym.make('CartPole-v0'); env.observation_space.dtype = np.dtype('float64')
    import random_env as env_; env = env_.RandomEnv(True)
    env = AsyncWrapperEnv(env, 0, 1.0, False)
    obs, info = env.reset()
    # print("main reset", obs)
    if hasattr(env,'np_struc'):
        action = np.random.randint(32, size=env.action_dtype.itemsize, dtype=np.uint8)
        action = np.frombuffer(action, dtype=env.action_dtype)
    else:
        action = env.action_space.sample()
    # print("main action", action)
    obs, reward, terminated, truncated, info = env.step(action)
    # print("main step ", obs)
    env.close()
    print("done")
