import threading, time
import numpy as np
# np.set_printoptions(precision=8, suppress=True)
import tensorflow as tf
import gym
import gym_trader

# maybe use MS-COCO to train both images and text captions, https://www.tensorflow.org/tutorials/text/image_captioning


# class DreamerAgent(object):
#     def __init__(self, action_space):
#         self.action_space = action_space
#     def get_action(self, observation):
#         # print("test observation {} shape {} dtype {}\n{}".format(type(observation), observation.shape, observation.dtype, observation))
#         return self.action_space.sample()

# TODO use A2C agent as base
class DreamerAgent(object):
    def __init__(self, env):
        self.env = env
    def get_action(self, observation):
        observation = gym.spaces.flatten(self.env.observation_space, observation)
        print("agent: observation {} shape {} dtype {}\n{}".format(type(observation), observation.shape, observation.dtype, observation))
        return env.action_space.sample()
    def save(self, path):
        pass





if __name__ == '__main__':
    me = 0
    # env = gym.make('FrozenLake-v0')
    # env = gym.make('CartPole-v0')
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    env = gym.make('Trader-v0', agent_id=me)
    env.seed(0)
    
    agent = DreamerAgent(env)
    for i_episode in range(2):
        reward_total = 0.0
        observation = env.reset()
        print("{}\n".format(observation))
        # env.render()
        for t_timesteps in range(3):
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            reward_total += reward

            # print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), observation))
            # env.render()
            time.sleep(1.01)
            
            if done: break
        print("agent: episode {}{} | timesteps {} | reward mean {} total {}\n".format(i_episode+1, (' DONE' if done else ''), t_timesteps+1, reward_total/(t_timesteps+1), reward_total))

    env.close()
    agent.save("")
