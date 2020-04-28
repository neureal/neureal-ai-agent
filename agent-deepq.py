import numpy as np
# np.set_printoptions(precision=8, suppress=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading, time, os
import gym
from baselines import deepq
import gym_trader

curdir = os.path.expanduser("~")
model_name = "gym-trader-deepq-256-2x64"
# model_name = "gym-cartpole-deepq-1024,512,256-4x256"
model_file = "{}/{}.pkl".format(curdir, model_name)

def callback(lcl, _glb):
    # print("{}".format(lcl))

    print("t {} reset? {} rew {:.18f} episode_rewards {:.18f} saved_mean_reward {}".format(lcl['t'], lcl['reset'], lcl['rew'] if 'rew' in lcl else 0.0, lcl['episode_rewards'][0], lcl['saved_mean_reward']))
    # if lcl['reset']: print("t {} episode_rewards {} saved_mean_reward {}".format(lcl['t'], lcl['episode_rewards'], lcl['saved_mean_reward']))
    # if if 'rew' in lcl and lcl['rew']: print("t {} episode_rewards {} saved_mean_reward {}".format(lcl['t'], lcl['episode_rewards'], lcl['saved_mean_reward']))

    # print("sum {}".format((sum(lcl['episode_rewards'][-101:-1]) / 100)))
    # is_solved = lcl['t'] > 5 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= max_reward
    # return is_solved

    return False

if __name__ == '__main__':
    # env = gym.make("CartPole-v0")
    env = gym.make("Trader-v0", agent_id=0)

    with tf.device("GPU:0"):
    # with tf.device("CPU:0"):
        model = deepq.learn(
            env,
            network='mlp',
            # lr=1e-3,
            # total_timesteps=10000,
            # buffer_size=5000,
            # learning_starts=100,
            # exploration_fraction=0.2,
            # exploration_final_eps=0.02,
            # prioritized_replay=True,
            # # param_noise=False,
            # target_network_update_freq=100,
            # print_freq=1,
            # # callback=callback,
            # checkpoint_freq = 200,
            load_path=model_file,
            # hiddens=[1024,512,256],
            # num_layers=4,
            # num_hidden=256,
            # # activation=tf.tanh
        )

        print("Saving model to {}".format(model_file))
        # model.save(model_file)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, model_file, max_to_keep=None)
        manager.save() # saves to directory


    # model = deepq.learn(env, network='mlp', total_timesteps=0, load_path=model_file)
    # while True:
    #     obs, done = env.reset(), False
    #     episode_rew = 0
    #     while not done:
    #         obs = np.expand_dims(np.array(obs), axis=0)
    #         # action = act(obs[None])[0]
    #         action = model.step(obs)[0].numpy()[0]
    #         obs, rew, done, _ = env.step(action)
    #         episode_rew += rew
    #         env.render()
    #     print("Episode reward", episode_rew)
    

    env.close()





# import gym
# import itertools
# import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.layers as layers

# import baselines.common.tf_util as U

# from baselines import logger
# from baselines import deepq
# from baselines.deepq.replay_buffer import ReplayBuffer
# from baselines.deepq.utils import ObservationInput
# from baselines.common.schedules import LinearSchedule

# import gym_trader


# def model(inpt, num_actions, scope, reuse=False):
#     """This model takes as input an observation and returns values of all actions."""
#     with tf.variable_scope(scope, reuse=reuse):
#         out = inpt
#         out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
#         out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
#         return out


# if __name__ == '__main__':
#     with U.make_session(num_cpu=2):
#         # Create the environment
#         # env = gym.make("CartPole-v0")
#         env = gym.make("Trader-v0", agent=0)
#         # Create all the functions necessary to train the model
#         act, train, update_target, debug = deepq.build_train(
#             make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
#             q_func=model,
#             num_actions=env.action_space.n,
#             optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
#         )
#         # Create the replay buffer
#         replay_buffer = ReplayBuffer(50000)
#         # Create the schedule for exploration starting from 1 (every action is random) down to
#         # 0.02 (98% of actions are selected according to values predicted by the model).
#         exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

#         # Initialize the parameters and copy them to the target network.
#         U.initialize()
#         update_target()

#         episode_rewards = [0.0]
#         obs = env.reset()
#         for t in itertools.count():
#             # Take action and update exploration to the newest value
#             action = act(obs[None], update_eps=exploration.value(t))[0]
#             new_obs, rew, done, _ = env.step(action)
#             # Store transition in the replay buffer.
#             replay_buffer.add(obs, action, rew, new_obs, float(done))
#             obs = new_obs

#             episode_rewards[-1] += rew
#             if done:
#                 obs = env.reset()
#                 episode_rewards.append(0)

#             is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
#             if is_solved:
#                 # Show off the result
#                 env.render()
#             else:
#                 # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
#                 if t > 1000:
#                     obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
#                     train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
#                 # Update target network periodically.
#                 if t % 1000 == 0:
#                     update_target()

#             if done and len(episode_rewards) % 10 == 0:
#                 logger.record_tabular("steps", t)
#                 logger.record_tabular("episodes", len(episode_rewards))
#                 logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
#                 logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
#                 logger.dump_tabular()
