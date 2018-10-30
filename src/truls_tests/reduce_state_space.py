import logging
import gym
import gym_super_mario_bros
import numpy as np
import sys
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from tensorforce.agents import PPOAgent
from movements import right_movements
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading


stdout_log_handler = logging.StreamHandler(sys.stdout)
stdout_log_handler.setFormatter(logging.Formatter('%(name)s: [%(levelname)s] %(message)s'))
root_log = logging.getLogger()
root_log.setLevel(logging.DEBUG)
root_log.addHandler(stdout_log_handler)
log = logging.getLogger('MLProject')

gym.logger.setLevel(gym.logger.DEBUG)

# env = gym_super_mario_bros.make('SuperMarioBros-v3')
_env = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=4, rom_mode='rectangle')
env = BinarySpaceToDiscreteSpaceEnv(_env, right_movements)

agent = PPOAgent(
    states=dict(type='float', shape=(20, 11, 3)),
    actions=dict(type='int', num_actions=len(right_movements)),
    network=[
        dict(type='flatten'),
        dict(type='dense', size=32),
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batching_capacity=5,
    step_optimizer=dict(
        type='adam',
        learning_rate=0.01
    )
)

done = True
state = np.empty((240, 256, 3))
state_downscaled = None


def anim_thread():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(_):
        if state_downscaled is not None:
            ax.imshow(state_downscaled)

    anim = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()


threading.Thread(target=anim_thread).start()

for step in range(500000):
    if done:
        state = env.reset()
    # state.shape = 240, 256, 3
    state_cutted = state[:, 85:215]
    state_downscaled = state_cutted[6::12, 6::12]
    action = agent.act(state_downscaled)
    state, reward, done, info = env.step(action)

    # Train the agent model
    agent.observe(reward=reward, terminal=False)

    if step % 100 == 0:
        log.debug('state {}: %s'.format(type(state)), state.shape)
        log.debug('reward {}: %s'.format(type(reward)), reward)
        log.debug('done {}: %s'.format(type(done)), done)
        log.debug('info {}: %s'.format(type(info)), info)
        log.debug('_y_pos {}: %s'.format(type(_env._y_position)), _env._y_position)

    env.render()

env.close()
