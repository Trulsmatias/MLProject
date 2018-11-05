import logging
import time
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from generations import Individual, Generation


def _vectofixedstr(vec, presicion=8):
    ret = []
    for el in vec:
        ret.append('{:.{}f}'.format(el, presicion))
    return '[' + ' '.join(ret) + ']'


class Simulator:
    def __init__(self, movements, max_steps):
        """
        Creates a new Simulator.
        The Simulator lets individuals play the game and assigns their resulting fitness to them.
        :param movements: a list of movements the individuals are allowed to make
        :param max_steps: the maximum number of simulation steps an individual is allowed to use
        """
        self.movements = movements
        self.max_steps = max_steps

        # TODO maybe another name on "env_expanded"?
        self.env_expanded = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=4, rom_mode='rectangle')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env_expanded, self.movements)

        self._log = logging.getLogger('MLProject')

    def _simulate_individual(self, individual: Individual, render):
        """
        Simulates a single individual and assigns its fitness score.
        This involves letting the individual play a game of Mario,
        and assigning the resulting fitness to the individual.
        :param individual:
        """
        state = self.env.reset()
        x_pos = 0
        reward_final = 0
        died = False

        last_fps_time = time.time()
        frames = 0
        for step in range(self.max_steps):
            # state.shape: 240/20 = 12, 256/21 = 12.19, 3
            state_cutted = state[6 * 12:18 * 12, 8 * 12:]  # 12 px per square. May cut in front of mario in the future
            state_downscaled = state_cutted[6::12, 6::12]
            self.state_downscaled = state_downscaled
            action = individual.agent.act(state_downscaled)
            # print('\r', _vectofixedstr(action, 12), end=' ')
            action = np.argmax(action)
            # print('taking action', self.movements[action], end='', flush=True)

            state, reward, done, info = self.env.step(action)
            x_pos = info['x_pos']
            reward_final += reward

            if render:
                self.env.render()

            if info["life"] <= 2:
                died = True
                break

            now = time.time()
            frames += 1
            if now - last_fps_time >= 1:
                fps = frames / (now - last_fps_time)
                # self._log.debug('FPS: {}'.format(fps))
                last_fps_time = now
                frames = 0

        fps = frames / (time.time() - last_fps_time)
        # self._log.debug('FPS: {}'.format(fps))

        # individual.fitness = x_pos
        individual.fitness = reward_final  # TODO: is acumulated reward the best fitnes function?

        if died:
            self._log.debug('Individual {} died. It achieved fitness {}'.format(individual.id, individual.fitness))
        else:
            self._log.debug(
                'Individual {} ran out of simulation steps. It achieved fitness {}'.format(individual.id,
                                                                                           individual.fitness))

    def simulate_generation(self, generation: Generation, render=False):
        """
        Simulates the whole generation and assigns each individual a fitness score.
        :param generation:
        :param render:
        """
        for individual in generation.individuals:
            self._simulate_individual(individual, render)
        print("\n")
