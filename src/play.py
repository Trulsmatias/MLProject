import logging
import time
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from generations import Individual, Generation
from pre_pro import get_sensor_map


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
        self.env_expanded = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='vanilla')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env_expanded, self.movements)
        # self.env.metadata['video.frames_per_second'] = 120
        # self.env_expanded.metadata['video.frames_per_second'] = 120

        self._log = logging.getLogger('MLProject.Simulator')

    def _simulate_individual(self, individual: Individual, render):
        """
        Simulates a single individual and assigns its fitness score.
        This involves letting the individual play a game of Mario,
        and assigning the resulting fitness to the individual.
        :param individual:
        """
        state = self.env.reset()

        x_pos = 0
        last_x_pos = 0
        reward_final = 0
        accumulated_fitness = 0
        died = False

        last_fps_time = time.time()
        frames = 0
        steps_standing_still = 0
        number_of_steps_standing_still_before_kill = 200

        for step in range(self.max_steps):
            self.state_downscaled = get_sensor_map(self.env_expanded)

            action = individual.agent.act(self.state_downscaled)
            # print('\r', _vectofixedstr(action, 12), end=' ')
            action = np.argmax(action)

            state, reward, done, info = self.env.step(action)

            if info['flag_get']:
                accumulated_fitness += x_pos

            x_pos = info['x_pos'] + accumulated_fitness

            reward_final += reward

            # Checks if reward is 0 to see if Mario stood still in the last step

            if last_x_pos -1 <= x_pos <= last_x_pos + 1:
                steps_standing_still += 1
                if steps_standing_still >= number_of_steps_standing_still_before_kill:
                    break
            else:
                steps_standing_still = 0

            last_x_pos = x_pos

            if render:
                self.env.render()

            if info["life"] <= 2:
                died = True
                break
 
            # now = time.time()
            frames += 1
            """
            if now - last_fps_time >= 1:
                fps = frames / (now - last_fps_time)
                self._log.debug('FPS: {}'.format(fps))
                last_fps_time = now
                frames = 0
            """

        fps = frames / (time.time() - last_fps_time)
        self._log.debug('Steps per second: {:.2f}'.format(fps))

        individual.fitness = x_pos
        # individual.fitness = reward_final

        if died:
            self._log.debug('Individual {} died. It achieved fitness {}'.format(individual.id, individual.fitness))
        else:
            self._log.debug(
                'Individual {} ran out of simulation steps. It achieved fitness {}'.format(individual.id,
                                                                                           individual.fitness))

    def simulate_generation(self, generation: Generation, render=True):
        """
        Simulates the whole generation and assigns each individual a fitness score.
        :param generation:
        :param render:
        """
        for individual in generation.individuals:
            self._simulate_individual(individual, render)

    def shutdown(self):
        """
        Does nothing. Needed for compatibility with ParallelSimulator
        """
        pass
