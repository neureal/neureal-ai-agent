import time, ctypes
import multiprocessing as mp
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gym
import model_util as util


class AsyncWrapperEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env, env_clock, render):
        super(AsyncWrapperEnv, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

        self._env_clock, self._render = env_clock, render
        # self._lock_print = mp.Lock()
        self._proc_ctrl = mp.Value('b', 0) # 1 = close, 0 = reset, -1 = step, -2 = done

        idx = 0; _action_idxs = util.gym_space_to_bytes(self.action_space.sample(), self.action_space)
        for i in range(len(_action_idxs)): idx += _action_idxs[i].size; _action_idxs[i] = idx
        _action_idxs = [0] + _action_idxs

        idx = 0; _obs_idxs = util.gym_space_to_bytes(self.observation_space.sample(), self.observation_space)
        _rew_done_zero = [np.frombuffer(np.asarray(0, np.float64), dtype=np.uint8), np.frombuffer(np.asarray(False, np.bool), dtype=np.uint8)]
        _obs_idxs += _rew_done_zero
        for i in range(len(_obs_idxs)): idx += _obs_idxs[i].size; _obs_idxs[i] = idx
        _obs_idxs = [0] + _obs_idxs

        self._action_idxs, self._obs_idxs, self._rew_done_zero = _action_idxs, _obs_idxs, _rew_done_zero

        self._action_shared = mp.sharedctypes.Array(ctypes.c_uint8, _action_idxs[-1])
        self._obs_shared = mp.sharedctypes.Array(ctypes.c_uint8, _obs_idxs[-1])

        self._proc = mp.Process(target=self._proc_run, name='ENV', args=())


    def _proc_run(self):
        action_view = np.asarray(self._action_shared.get_obj())
        obs_view = np.asarray(self._obs_shared.get_obj())
        action = np.zeros(action_view.shape, action_view.dtype)

        while self._proc_ctrl.value != 1:

            if self._proc_ctrl.value == 0:
                obs = self.env.reset()
                # print("proc reset", obs)
                obs = util.gym_space_to_bytes(obs, self.observation_space)
                obs += self._rew_done_zero
                obs = np.concatenate(obs)
                with self._obs_shared.get_lock():
                    if not np.array_equal(obs_view, obs): np.copyto(obs_view, obs, casting='no')
                self._proc_ctrl.value = -1

            elif self._proc_ctrl.value == -1:
                with self._action_shared.get_lock():
                    if not np.array_equal(action_view, action): np.copyto(action, action_view, casting='no')
                action_space = util.gym_bytes_to_space(action, self.action_space, self._action_idxs, [0])

                obs, reward, done, _ = self.env.step(action_space)
                # print("proc step", obs)
                obs = util.gym_space_to_bytes(obs, self.observation_space)
                obs += [np.frombuffer(np.asarray(reward, np.float64), dtype=np.uint8), np.frombuffer(np.asarray(done, np.bool), dtype=np.uint8)]
                obs = np.concatenate(obs)
                with self._obs_shared.get_lock():
                    if not np.array_equal(obs_view, obs): np.copyto(obs_view, obs, casting='no')
                if done: self._proc_ctrl.value = -2

            if self._render: self.env.render()
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
        while self._proc_ctrl.value != -1: time.sleep(0)

        with self._obs_shared.get_lock(): np.copyto(obs, obs_view, casting='no')
        obs = util.gym_bytes_to_space(obs, self.observation_space, self._obs_idxs, [0])
        return obs

    def step(self, action):
        action_view = np.asarray(self._action_shared.get_obj())
        obs_view = np.asarray(self._obs_shared.get_obj())
        obs = np.zeros(obs_view.shape, obs_view.dtype)

        action = util.gym_space_to_bytes(action, self.action_space)
        action = np.concatenate(action)
        with self._action_shared.get_lock():
            if not np.array_equal(action_view, action): np.copyto(action_view, action, casting='no')
        time.sleep(0)

        with self._obs_shared.get_lock(): np.copyto(obs, obs_view, casting='no')
        reward = obs[self._obs_idxs[-3]:self._obs_idxs[-2]]
        reward = np.frombuffer(reward, dtype=np.float64).item()
        done = obs[self._obs_idxs[-2]:self._obs_idxs[-1]]
        done = np.frombuffer(done, dtype=np.bool).item()
        obs = util.gym_bytes_to_space(obs, self.observation_space, self._obs_idxs, [0])
        return obs, reward, done, {}

if __name__ == '__main__':
    ## test
    # env = gym.make('CartPole-v0'); env.observation_space.dtype = np.dtype('float64')
    import random_env as env_; env = env_.RandomEnv()
    env = AsyncWrapperEnv(env, 0.003, False)
    obs = env.reset()
    print("main reset", obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("main step", obs)
    env.close()
    print("done")
