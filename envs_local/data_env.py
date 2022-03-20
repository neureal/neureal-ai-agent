from collections import OrderedDict
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import tensorflow_datasets as tfds
import gym
import gym_util


class DataEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data_src):
        super(DataEnv, self).__init__()
        self.data_src = data_src

        if data_src == 'shkspr':
            ds = tfds.as_numpy(tfds.load('tiny_shakespeare', batch_size=-1)) # \n = done
            ds = ds['train']['text'][0]
            ds = np.frombuffer(ds, np.uint8)
            # done = np.frombuffer(b'.\n', np.uint8)
            # ds = ds[ds!=done[1]] # take out newlines
            # split = np.asarray(np.nonzero(ds==done[0])[0])+1 # 6960
            # ds = ds[:split[-1]]
            ds = ds[:,None]

            # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            space = gym.spaces.Dict()
            # space.spaces['timestamp'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
            # space.spaces['data'] = gym.spaces.Discrete(256) # np.int64
            space.spaces['data'] = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            # space.spaces['data'] = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8) # combine to latent
            self.observation_space = space

            space = gym.spaces.Dict()
            space.spaces['data'] = gym.spaces.Discrete(256) # np.int64
            # space.spaces['data'] = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            self.action_space = space

            self.reward_range = (0.0,1.0)

        # TODO split (reshape into batch) image into blocks or pixels to test for spatial autoregression
        # if data_src == 'mnist':
        #     ds = tfds.as_numpy(tfds.load('mnist', batch_size=-1))
        #     # self.dsl = ds['train']['label'][:,None]
        #     ds = ds['train']['image']

        #     # train_obs, test_obs = tf.image.resize(train_obs, (16,16), method='nearest').numpy(), tf.image.resize(test_obs, (16,16), method='nearest').numpy()
        #     # self.action_space = gym.spaces.Discrete(10)
        #     # self.observation_space = gym.spaces.Box(low=0, high=255, shape=list(ds.shape)[1:], dtype=np.uint8)

        #     self.action_space = gym.spaces.Discrete(256)
        #     self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        #     self.reward_range = (0.0,1.0)

        #     self.pxl_x, self.pxl_y, self.x_max, self.y_max = 0, 0, ds.shape[1], ds.shape[2]
        # if data_src == 'mnist-mv':
        #     ds = tfds.as_numpy(tfds.load('moving_mnist', batch_size=-1))
        #     ds = ds['test']['image_sequence'].reshape((200000,64,64,1))

        # ds = ds[:16]
        self.ds, self.ds_idx, self.ds_max = ds, 0, 64

        self.action_zero = gym_util.get_space_zero(self.action_space)
        self.obs_zero = gym_util.get_space_zero(self.observation_space)
        self.state = self.action_zero, self.obs_zero, np.float64(0.0), False, {}
        self.item_accu = []
        self.episode = 0
        

    def step(self, action): return self._request(action)
    def reset(self): return self._request(None)[0]
    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.state
        # if action is None: print("{}\n".format(obs))
        # else: print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))
        if action is None:
            if self.data_src == 'shkspr':
                text = np.asarray(self.item_accu, dtype=np.uint8)
                text = text.tobytes()
                try: text = text.decode('utf-8')
                except: pass
                print("\n\n-----------------------------------------------------------------------------------------------------------------")
                print(text)
            self.item_accu = []
        else:
            self.item_accu.append(action)

    def _request(self, action):
        reward, done, info = np.float64(0.0), False, {}
        # obs = self.observation_space.sample()
        # reward = np.float64(np.random.standard_normal())

        # obs = {'timestamp':np.asarray([self.ds_idx], np.float64), 'data':self.ds[self.ds_idx]}
        obs = OrderedDict()
        # obs['timestamp'] = np.asarray([self.ds_idx], self.observation_space['timestamp'].dtype)
        obs['data'] = np.asarray(self.ds[self.ds_idx], self.observation_space['data'].dtype)
        # latent = np.concatenate([self.ds[self.ds_idx]])
        # latent = np.concatenate([self.ds[self.ds_idx],[self.ds_idx]]) # combine to latent
        # obs['data'] = np.asarray(latent, self.observation_space['data'].dtype) # combine to latent
        if self.data_src == 'shkspr':
            if action is not None: # predict next byte
                # obs_prev = self.ds[self.ds_idx-1]
                action = action['data'][0] if isinstance(action['data'], np.ndarray) else action['data']
                target = obs['data'][0] if isinstance(obs['data'], np.ndarray) else obs['data']
                # if action >= 122: print(action, self.episode)
                if action == target: reward = np.float64(1.0)
            else: self.episode += 1
            self.ds_idx += 1
            if self.ds_idx >= self.ds_max:
                done = True
        # if self.data_src == 'mnist':
        #     obs = obs[self.pxl_x, self.pxl_y]
        #     if action is not None:
        #         # action = np.asarray([action], self.dsl.dtype)
        #         # if action == self.dsl[self.ds_idx-1]: reward = np.float64(1.0)
        #         action = np.asarray([action], np.uint8)
        #         if action == obs: reward = np.float64(1.0)
        #     # TODO add ds_idx, pxl_x, pxl_y to obs
        #     self.pxl_x += 1
        #     if self.pxl_x >= self.x_max:
        #         self.pxl_x = 0; self.pxl_y += 1
        #         if self.pxl_y >= self.y_max:
        #             self.pxl_y = 0; self.ds_idx += 1; done = True

        if self.ds_idx >= self.ds_max:
            self.ds_idx = 0


        self.state = (action, obs, reward, done, info)
        return obs, reward, done, info

if __name__ == '__main__':
    ## test
    env = DataEnv('shkspr')
    obs = env.reset()
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
