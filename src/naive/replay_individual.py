import os
import time
import gym_super_mario_bros
import numpy as np
from keras.utils import h5dict
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from naive.agent import NNAgent
from naive.generations import Individual
from naive.movements import right_movements
from naive.pre_pro import get_sensor_map


def vectofixedstr(vec, presicion=8):
    ret = []
    for el in vec:
        ret.append('{:.{}f}'.format(el, presicion))
    return '[' + ' '.join(ret) + ']'


def replay_individual(individual, movements):
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
            action = individual.agent.act(state_downscaled)

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
                # print('FPS: {:.3f}'.format(fps))
                last_fps_time = now
                frames = 0
        else:
            time.sleep(0.001)

    env.close()


if __name__ == '__main__':
    workdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    h5mod = h5dict(os.path.join(workdir, 'saved_data', 'best', 'module_gen47.h5'))

    weights = [h5mod['model_weights']['dense']['dense']['kernel:0'],
               h5mod['model_weights']['dense']['dense']['bias:0'],
               h5mod['model_weights']['dense_1']['dense_1']['kernel:0'],
               h5mod['model_weights']['dense_1']['dense_1']['bias:0']]
    print([np.shape(w) for w in weights])

    individual = Individual(NNAgent(state_space_shape=(13, 10), action_space_size=len(right_movements)))
    individual.agent.set_weights(weights)

    replay_individual(individual, right_movements)

    """
    print([key for key in h5mod['model_weights']])
    print([key for key in h5mod['model_weights']['dense']])
    print([key for key in h5mod['model_weights']['dense']['dense']])
    print(np.shape(h5mod['model_weights']['dense']['dense']['kernel:0']))
    print(np.shape(h5mod['model_weights']['dense']['dense']['bias:0']))

    print([key for key in h5mod['model_weights']['dense_1']])
    print([key for key in h5mod['model_weights']['dense_1']['dense_1']])
    print(np.shape(h5mod['model_weights']['dense_1']['dense_1']['kernel:0']))
    print(np.shape(h5mod['model_weights']['dense_1']['dense_1']['bias:0']))

    print([key for key in h5mod['model_weights']['activation']])
    print([key for key in h5mod['model_weights']['reshape']])
    """

    # replay_individual_from_model('/home/magnus/model_fitness2226.h5')
