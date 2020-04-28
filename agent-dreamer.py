import threading, time
import numpy as np
# np.set_printoptions(precision=8, suppress=True)
import tensorflow as tf
import gym
import gym_trader

# maybe use MS-COCO to train both images and text captions, https://www.tensorflow.org/tutorials/text/image_captioning


# TODO use A2C agent as base
class DreamerAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    def get_action(self, observation):
        # print("test observation {} shape {} dtype {}\n{}".format(type(observation), observation.shape, observation.dtype, observation))
        return self.action_space.sample()



if __name__ == '__main__':
    me = 0
    # env = gym.make('FrozenLake-v0')
    # env = gym.make('CartPole-v0')
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    env = gym.make('Trader-v0', agent_id=me)
    env.seed(0)

    agent = DreamerAgent(env.action_space)
    for i_episode in range(2):
        observation = env.reset()
        env.render()
        for t_timesteps in range(3):
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)

            env.render()
            # print("{} {}\nreward {:.18f} done? {}\n".format(action, observation, reward, done))
            time.sleep(1.01)
            
            if done: break
        print("agent: episode {} finished after {} timesteps\n".format(i_episode+1 , t_timesteps+1))

    env.close()
