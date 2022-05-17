
import os
import sys
print(sys.path)
import numpy as np

from unstable_baselines.envs.olympics_integrated.chooseenv import make
from unstable_baselines.common.env_wrapper import JidiFlattenEnv


def get_two_agent_action():
    action = []
    action.append(np.random.rand(2) * 100)
    action.append(np.random.rand(2) * 100)
    return action


def test_make_env():    
    env = make('olympics-integrated')
    state = env.reset()
    print(state)
    action_temp = [[[0], [0]], [[0], [0]]]
    next_state, reward, done, info_before, info_after = env.step(action_temp)
    print("\033[32m next_state\033[0m", next_state)
    print("\033[32m reward\033[0m", reward)
    print("\033[32m done\033[0m", done)
    print("\033[32m info_before\033[0m", info_before)
    print("\033[32m info_after\033[0m", info_after)
    
    img_obs = next_state[0]['obs']['agent_obs']
    print(img_obs.min(), img_obs.max())
    print(type(img_obs), img_obs.shape)

    return env


def test_flatten_wrapper():
    env = make('olympics-integrated')
    env = JidiFlattenEnv(env)
    obs = env.reset()
    print(obs)






# test_make_env()
test_flatten_wrapper()


# 测试传统gym的设置

import gym

# atari = gym.make("Alien-v0")
# atari = gym.make("MontezumaRevenge-v0")
# obs = atari.reset()
# print("atari obs", obs.shape)

# print(atari.observation_space)
# print(atari.action_space)

mujoco = gym.make("HalfCheetah-v2")
obs = mujoco.reset()
print("mujoco的状态有没有batch维度", obs.shape)
print(mujoco.observation_space)
print(mujoco.action_space)