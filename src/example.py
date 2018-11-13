from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
from src import pre_pro
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
        print(np.shape(state))
    state, reward, done, info = env.step(env.action_space.sample())

    if step % 100 == 0:
        pre_pro.simple_prepro(state)

    env.render()

env.close()