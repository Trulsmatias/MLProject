from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

from tensorforce.agents import PPOAgent

agent = PPOAgent(
    states=dict(type='int', shape=(240, 256, 3)),
    actions=dict(type='int', num_actions=7),
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

done = True
action = env.action_space.sample()
print(action)
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(action)  #env.action_space.sample()  # 0-6
    agent.observe(reward=reward, terminal=False)
    action = agent.act(state)

    env.render()

env.close()