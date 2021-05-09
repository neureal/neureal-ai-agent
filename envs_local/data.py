import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import tensorflow_datasets as tfds
import gym
import model_util as util


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

            obs_space_dict = gym.spaces.Dict()
            obs_space_dict.spaces['timestamp'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
            obs_space_dict.spaces['data'] = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            self.observation_space = obs_space_dict

            # action_space_dict = gym.spaces.Dict()
            # action_space_dict.spaces['pair'] = gym.spaces.Discrete(self.trade_actions) # trade pair (BTC<>USD) + NOOP
            # action_space_dict.spaces['percent'] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64) # percent of available margin

            # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(256)
            self.reward_range = (0.0,1.0)
        # TODO split (reshape into batch) image into blocks or pixels to test for spatial autoregression
        # if data_src == 'mnist':
        #     ds = tfds.as_numpy(tfds.load('mnist', batch_size=-1))
        #     # self.dsl = ds['train']['label'][:,None]
        #     ds = ds['train']['image']

        #     # train_obs, test_obs = tf.image.resize(train_obs, (16,16), method='nearest').numpy(), tf.image.resize(test_obs, (16,16), method='nearest').numpy()
        #     # self.observation_space = gym.spaces.Box(low=0, high=255, shape=list(ds.shape)[1:], dtype=np.uint8)
        #     # self.action_space = gym.spaces.Discrete(10)

        #     self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        #     self.action_space = gym.spaces.Discrete(256)
        #     self.reward_range = (0.0,1.0)

        #     self.pxl_x, self.pxl_y, self.x_max, self.y_max = 0, 0, ds.shape[1], ds.shape[2]
        # if data_src == 'mnist-mv':
        #     ds = tfds.as_numpy(tfds.load('moving_mnist', batch_size=-1))
        #     ds = ds['test']['image_sequence'].reshape((200000,64,64,1))

        # ds = ds[:16]
        self.ds, self.ds_idx, self.ds_max = ds, 0, 64

        self.ob_spec, self.ob_zero, _ = util.gym_get_spec(self.observation_space)
        self.action_spec, self.action_zero, _ = util.gym_get_spec(self.action_space)
        self.state = self.action_zero, self.ob_zero, np.float64(0.0), False, {}
        self.item_accu = []
        

    def step(self, action): return self._request(action)
    def reset(self): return self._request(None)[0]
    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.state
        # if action == None: print("{}\n".format(obs))
        # else: print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))
        if action == None:
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

        obs = {'timestamp':np.asarray([self.ds_idx], self.ob_spec['timestamp']['dtype']), 'data':self.ds[self.ds_idx]}
        if self.data_src == 'shkspr':
            if action is not None:
                # obs_prev = self.ds[self.ds_idx-1]
                action_pred = np.asarray([action], self.ob_spec['data']['dtype'])
                if action_pred == obs['data']: reward = np.float64(1.0)
            # TODO add ds_idx to obs
            self.ds_idx += 1
            if self.ds_idx >= self.ds_max:
                done = True
        # if self.data_src == 'mnist':
        #     obs = obs[self.pxl_x, self.pxl_y]
        #     if action is not None:
        #         # action_pred = np.asarray([action], self.dsl.dtype)
        #         # if action_pred == self.dsl[self.ds_idx-1]: reward = np.float64(1.0)
        #         action_pred = np.asarray([action], np.uint8)
        #         if action_pred == obs: reward = np.float64(1.0)
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
