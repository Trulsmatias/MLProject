import pickle
import time
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from movements import right_movements
from pre_pro import get_sensor_map
from util import get_path_of


def vectofixedstr(vec, presicion=8):
    ret = []
    for el in vec:
        ret.append('{:.{}f}'.format(el, presicion))
    return '[' + ' '.join(ret) + ']'


def replay_genome(genome, movements):
    env_expanded = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='vanilla')
    env = BinarySpaceToDiscreteSpaceEnv(env_expanded, movements)

    done = True
    unticked = 0
    tick_interval = 1 / 30
    last_tick_time = time.time()

    fps = 0
    frames = 0
    last_fps_time = time.time()

    for _ in range(500000):

        unticked += time.time() - last_tick_time
        last_tick_time = time.time()
        ticked = False

        while unticked >= tick_interval:
            if done:
                state = env.reset()

            state_downscaled = get_sensor_map(env_expanded)
            action = genome.calculate_action(state_downscaled)

            print('\rFPS: {:.3f}'.format(fps), end=' ')
            print(vectofixedstr(action, 10), end=' ')
            action = np.argmax(action)
            print('taking action', movements[action], end='', flush=True)

            state, reward, done, info = env.step(action)

            env.render()

            ticked = True
            frames += 1
            unticked -= tick_interval

        if ticked:
            now = time.time()
            if now - last_fps_time >= 1:
                fps = frames / (now - last_fps_time)
                last_fps_time = now
                frames = 0
        else:
            time.sleep(0.001)

    env.close()


if __name__ == '__main__':
    filename = get_path_of('saved_data/best/model_gen63.obj')

    with open(filename, 'rb') as file:
        genome = pickle.load(file)

    replay_genome(genome, right_movements)
