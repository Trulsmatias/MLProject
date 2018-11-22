from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from naive.movements import right_movements as movements
from naive.pre_pro import get_sensor_map
import gym_super_mario_bros
import numpy as np


env_expanded = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='vanilla')
env = BinarySpaceToDiscreteSpaceEnv(env_expanded, movements)


def simulate_run(genome, max_steps, render):
    state = env.reset()
    fitness = 0
    highest_x_pos = 0
    accumulated_fitness = 0

    number_of_steps_standing_still_before_kill = 50

    for step in range(max_steps):
        state_downscaled = get_sensor_map(env_expanded)
        action = genome.calculate_action(state_downscaled)
        action = np.argmax(action)

        state, reward, done, info = env.step(action)

        if info['flag_get']:
            accumulated_fitness += fitness

        fitness = info['x_pos'] + accumulated_fitness - (400 - info['time'])

        if info['x_pos'] <= highest_x_pos:
            steps_standing_still += 1
            if steps_standing_still >= number_of_steps_standing_still_before_kill:
                break
        else:
            highest_x_pos = info['x_pos']
            steps_standing_still = 0


        if render:
            env.render()

        if info["life"] <= 2:
            died = True
            break

    genome.fitness = fitness


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    input_nodes = 7
    output_nodes = 5

    b = a[input_nodes:input_nodes+output_nodes]
    print(b)
