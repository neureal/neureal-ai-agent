import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import tensorflow_datasets as tfds
import gym


class DataEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data_src):
        super(DataEnv, self).__init__()
        self.data_src = data_src

        # TODO split (reshape into batch) image into blocks or pixels to test for spatial autoregression
        if data_src == 'mnist':
            ds = tfds.as_numpy(tfds.load('mnist', batch_size=-1))
            self.dsl = ds['train']['label'][:,None]
            ds = ds['train']['image']
            # train_obs, test_obs = tf.image.resize(train_obs, (16,16), method='nearest').numpy(), tf.image.resize(test_obs, (16,16), method='nearest').numpy()

            self.observation_space = gym.spaces.Box(low=0, high=255, shape=list(ds.shape)[1:], dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(10)
            self.reward_range = (0.0,1.0)
        # if data_src == 'mnist-mv':
        #     ds = tfds.as_numpy(tfds.load('moving_mnist', batch_size=-1))
        #     ds = ds['test']['image_sequence'].reshape((200000,64,64,1))
        if data_src == 'shkspr':
            ds = tfds.as_numpy(tfds.load('tiny_shakespeare', batch_size=-1)) # \n = done
            ds = ds['train']['text'][0]
            ds = np.frombuffer(ds, np.uint8)
            # done = np.frombuffer(b'.\n', np.uint8)
            # ds = ds[ds!=done[1]] # take out newlines
            # split = np.asarray(np.nonzero(ds==done[0])[0])+1 # 6960
            # ds = ds[:split[-1]]
            ds = ds[:,None]

            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(256)
            self.reward_range = (0.0,1.0)

        ds = ds[:128]
        self.ds, self.ds_idx = ds, 0

        self.action_noop = 0
        self.obs_space_zero = self.observation_space.sample()
        self.obs_space_zero.fill(0)
        self.state = self.action_noop, self.obs_space_zero, np.float64(0.0), False, {}
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
            self.item_accu.append(action.item())

    def _request(self, action):
        reward, done, info = np.float64(0.0), False, {}

        # obs = self.observation_space.sample()
        # reward = np.float64(np.random.standard_normal())

        obs = self.ds[self.ds_idx]
        if action is not None:
            if self.data_src == 'mnist':
                action_pred = np.asarray([action], self.dsl.dtype)
                if action_pred == self.dsl[self.ds_idx-1]: reward = np.float64(1.0)
            if self.data_src == 'shkspr':
                # obs_prev = self.ds[self.ds_idx-1]
                action_pred = np.asarray([action], self.observation_space.dtype)
                if action_pred == obs: reward = np.float64(1.0)

        self.ds_idx += 1
        if self.ds_idx >= len(self.ds) - 1:
            self.ds_idx = 0; done = True 
        
        # dsl = np.split(ds, split) # split on period into sentances
        # for obs in dsl:
        #     inputs['obs'] = tf.convert_to_tensor(obs[:,None])
        #     loss, outputs = model.TRANS_train(inputs)

        # dsrt = tf.RaggedTensor.from_row_limits(ds, row_limits=split)
        # dsrt = tf.expand_dims(dsrt, axis=2)

        self.state = (action, obs, reward, done, info)
        return obs, reward, done, info