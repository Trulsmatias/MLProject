from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from src.movements import right_movements
from src.pre_pro import get_sensor_map
import gym_super_mario_bros
import numpy as np


env_expanded = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='vanilla')
env = BinarySpaceToDiscreteSpaceEnv(env_expanded, right_movements)


def simulate_run(genome, max_steps, render):
    state = env.reset()
    x_pos = 0
    last_x_pos = 0
    accumulated_fitness = 0
    number_of_steps_standing_still_before_kill = 100

    for step in range(max_steps):
        state_downscaled = get_sensor_map(env_expanded)
        action = genome.calculate_action(state_downscaled, 130, 7)
        action = np.argmax(action)

        state, reward, done, info = env.step(action)

        if info['flag_get']:
            accumulated_fitness += x_pos

        x_pos = info['x_pos'] + accumulated_fitness

        if last_x_pos - 1 <= x_pos <= last_x_pos + 1:
            steps_standing_still += 1
            if steps_standing_still >= number_of_steps_standing_still_before_kill:
                break
        else:
            steps_standing_still = 0

        last_x_pos = x_pos

        if render:
            env.render()

        if info["life"] <= 2:
            died = True
            break

    genome.fitness = x_pos


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    input_nodes = 7
    output_nodes = 5

    b = a[input_nodes:input_nodes+output_nodes]
    print(b)
