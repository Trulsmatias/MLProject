import logging
import gym
import gym_super_mario_bros
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from tensorforce.agents import PPOAgent
from movements import basic_movements


print(os.environ['PATH'])
os.environ['PATH'] = os.path.join('C:\\Users\\Magnus\\PycharmProjects\\ML-prosjekt\\FCEUX')
logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler(sys.stdout)])
gym.logger.setLevel(gym.logger.DEBUG)


gym.envs.register(
    id='SuperMarioBros-1-1-v0',
    entry_point='ppaquette_gym_super_mario:MetaSuperMarioBrosEnv'
)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
#_env = gym_super_mario_bros.SuperMarioBrosEnv()
#print(type(_env))
#env = BinarySpaceToDiscreteSpaceEnv(_env, basic_movements)
#print(type(env))

env = gym.make('SuperMarioBros-1-1-v0')
print(type(env))
env.disable_out_pipe = True
env.disable_in_pipe = True

agent = PPOAgent(
    states=dict(type='float', shape=(10,)),
    actions=dict(type='int', num_actions=2),
    network=[
        dict(type='flatten'),
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batching_capacity=10,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

done = True
state = np.empty((240, 256, 3))

for step in range(5000):
    if done:
        state = env.reset()
        #plt.imshow(state[:, :, 2], cmap='gray')
        #plt.show()

    state = np.zeros((10,))
    action = agent.act(state)
    print(action)
    # action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    # Train the agent model
    # agent.observe(reward=reward, terminal=False)

    if step % 100 == 0:
        print('\nstate ({}):'.format(type(state)), state.shape)
        print('reward ({}):'.format(type(reward)), reward)
        print('done ({}):'.format(type(done)), done)
        print('info ({}):'.format(type(info)), info)
        #print('_y_pos ({}):'.format(type(_env._y_position)), _env._y_position)

    env.render()

env.close()
