import gym_super_mario_bros
import numpy as np
import matplotlib.pyplot as plt
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from tensorforce.agents import PPOAgent
from movements import basic_movements


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, basic_movements)

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
    agent.states
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

    env.render()

env.close()
