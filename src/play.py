import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from generations import Individual, Generation


def vectofixedstr(vec, presicion=8):
    ret = []
    for el in vec:
        ret.append('{:.{}f}'.format(el, presicion))
    return '[' + ' '.join(ret) + ']'


class Simulator:
    def __init__(self, movements, max_steps):
        self.movements = movements
        self.max_steps = max_steps

        self._env = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=4, rom_mode='rectangle')
        self.env = BinarySpaceToDiscreteSpaceEnv(self._env, self.movements)

    def _simulate_individual(self, individual: Individual):
        """
        Simulates a single individual and assigns its fitness score.
        This involves letting the individual play a game of Mario,
        and assigning the resulting fitness to the individual.
        :param individual:
        """
        done = False
        state = self.env.reset()
        x_pos = 0
        for step in range(self.max_steps):
            if done:
                break

            """if step % 100 == 0:
                print('Randomizing weights!')
                weights = individual.agent.model.get_weights()
                for i in range(len(weights)):
                    for index, w in np.ndenumerate(weights[i]):
                        weights[i][index] = np.random.random() * 2 - 1
                individual.agent.model.set_weights(weights)
            """

            state_downscaled = state[6::12, 6::12]
            action = individual.agent.act(state_downscaled)
            print('\r', action.shape, vectofixedstr(action, 12), end=' ')
            action = np.argmax(action)
            print('taking action', self.movements[action], end='', flush=True)

            state, reward, done, info = self.env.step(action)
            x_pos = info['x_pos']

            if step % 100 == 0:
                pass
                """log.debug('state {}: %s'.format(type(state)), state.shape)
                log.debug('reward {}: %s'.format(type(reward)), reward)
                log.debug('done {}: %s'.format(type(done)), done)
                log.debug('info {}: %s'.format(type(info)), info)
                log.debug('_y_pos {}: %s'.format(type(self._env._y_position)), self._env._y_position)"""

            self.env.render()

        individual.fitness = x_pos
        self.env.close()

    def simulate_generation(self, generation: Generation):
        """
        Simulates the whole generation and assigns each individual a fitness score.
        :param generation:
        """
        for individual in generation.individuals:
            self._simulate_individual(individual)
