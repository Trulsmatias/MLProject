import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from movements import basic_movements


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, basic_movements)

'''
agent = PPOAgent(
    states=dict(type='float', shape=(240, 256, 3)),
    actions=dict(type='int', num_actions=len(basic_movements)),
    network=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batching_capacity=1000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)
'''

done = True
state = np.empty((240, 256, 3))

for step in range(5000):
    if done:
        state = env.reset()

    # action = agent.act(state)
    action = env.action_space.sample()
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
