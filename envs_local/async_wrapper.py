import time, ctypes
import multiprocessing as mp
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gym


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

        # action = self.action_space.sample()
        # obs = self.observation_space.sample()
        # self._action_shared = mp.sharedctypes.Array('f', 1)
        # self._obs_shared = mp.sharedctypes.Array('f', 4)
        # self._action_shared = mp.sharedctypes.Array(self.action_space.dtype.str, self.action_space.shape)
        # self._obs_shared = mp.sharedctypes.Array(self.observation_space.dtype.str, self.observation_space.shape)
        self._action_shared = mp.sharedctypes.Array(ctypes.c_int64, 1)
        self._obs_shared = mp.sharedctypes.Array(ctypes.c_double, 4)
        self._reward_shared = mp.sharedctypes.Value(ctypes.c_double)
        self._done_shared = mp.sharedctypes.Value(ctypes.c_bool)

        self._proc = mp.Process(target=self._proc_run, name='ENV', args=())

    def _proc_run(self):
        action_view = np.asarray(self._action_shared.get_obj())
        obs_view = np.asarray(self._obs_shared.get_obj())
        
        action = np.zeros(action_view.shape, action_view.dtype)
        while self._proc_ctrl.value != 1:

            if self._proc_ctrl.value == 0:
                obs = self.env.reset()
                if self._render: self.env.render()
                with self._obs_shared.get_lock(): np.copyto(obs_view, obs, casting='no')
                self._proc_ctrl.value = -1
            
            if self._proc_ctrl.value == -1:
                with self._action_shared.get_lock():
                    if not np.array_equal(action_view, action):
                        np.copyto(action, action_view, casting='no')

                obs, reward, done, _ = self.env.step(action[0])
                if self._render: self.env.render()
                with self._obs_shared.get_lock(), self._reward_shared.get_lock(), self._done_shared.get_lock():
                    np.copyto(obs_view, obs, casting='no')
                    self._reward_shared.value = reward
                    self._done_shared.value = done
                if done and self._proc_ctrl.value != 0: self._proc_ctrl.value = -2

            time.sleep(self._env_clock)

        self.env.close()


    # def seed(self): return self.env.seed()
    # def close(self): return self.env.close()
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
        return obs

    def step(self, action):
        action_view = np.asarray(self._action_shared.get_obj())
        obs_view = np.asarray(self._obs_shared.get_obj())
        obs = np.zeros(obs_view.shape, obs_view.dtype)

        with self._action_shared.get_lock(): np.copyto(action_view, action, casting='no')
        time.sleep(0)
        with self._obs_shared.get_lock(), self._reward_shared.get_lock(), self._done_shared.get_lock():
            np.copyto(obs, obs_view, casting='no')
            reward = self._reward_shared.value
            done = self._done_shared.value

        return obs, reward, done, None
