import pickle
import time
import gym_super_mario_bros
import numpy as np
from scipy.misc import imsave
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from naive.movements import right_movements
from naive.pre_pro import get_sensor_map
import matplotlib.pyplot as plt
from neat.config import load_config
from util import get_path_of


def vectofixedstr(vec, presicion=8):
    ret = []
    for el in vec:
        ret.append('{:.{}f}'.format(el, presicion))
    return '[' + ' '.join(ret) + ']'


def replay_genome(genome, movements):
    env_expanded = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='vanilla')
    env = BinarySpaceToDiscreteSpaceEnv(env_expanded, movements)

    print('Number of genes: ', len(genome.connection_genes))
    for gene in genome.connection_genes:
        print(gene.in_node, gene.out_node, gene.weight, gene.innovation_number, gene.type, gene.enabled)

    done = True
    unticked = 0
    tick_interval = 1/30
    last_tick_time = time.time()

    fps = 0
    frames = 0
    last_fps_time = time.time()

    for _ in range(500000):

        unticked += time.time() - last_tick_time
        last_tick_time = time.time()
        ticked = False

        # while unticked >= tick_interval:
        if done:
            state = env.reset()

        state_downscaled = get_sensor_map(env_expanded)
        action = genome.calculate_action(state_downscaled)



        # print('\rFPS: {:.3f}'.format(fps), end=' ')
        # print(vectofixedstr(action, 10), end=' ')
        action = np.argmax(action)
        print('\rtaking action', movements[action], end='', flush=True)

        state, reward, done, info = env.step(action)

        # imsave('stateIMGS/mario' + str(_) + '.png', state)


        # make_controller(movements[action], _)



        env.render()

        if info["life"] <= 2:
            died = True
            break

        ticked = True
        frames += 1
        unticked -= tick_interval

        # if ticked:
        #     now = time.time()
        #     if now - last_fps_time >= 1:
        #         fps = frames / (now - last_fps_time)
        #         last_fps_time = now
        #         frames = 0
        # else:
        #     time.sleep(0.001)

    env.close()


def make_controller(game_inputs, step):
    controller = np.full((10, 18, 3), 255, dtype=np.int)

    LIGHT_RED = [255, 213, 213]
    DARK_RED = [196, 0, 0]

    GREY = [150, 150, 150]
    BLACK = [0, 0, 0]

    UP = [(2, 3),(2, 4),(3, 3),(3, 4)]
    DOWN = [(6, 3),(6, 4),(7, 3),(7, 4)]
    LEFT = [(4, 1),(4, 2),(5, 1),(5, 2)]
    RIGHT = [(4, 5),(4, 6),(5, 5),(5, 6)]

    A = [(5, 15),(5, 16),(6, 15),(6, 16)]
    B = [(5, 11),(5, 12),(6, 11),(6, 12)]

    inputs = [UP, DOWN, LEFT, RIGHT, A, B]

    for input in inputs:
        if input[0][1] < 9:
            register_input(controller, input, GREY)
        else:
            register_input(controller, input, LIGHT_RED)

    for input in game_inputs:
        if input == 'right':
            register_input(controller, RIGHT, BLACK)
        elif input == 'A':
            register_input(controller, A, DARK_RED)
        elif input == 'B':
            register_input(controller, B, DARK_RED)

    # plt.imshow(controller)

    imsave('controllerIMGS/testIMG'+str(step)+'.png' ,controller)

    # plt.show()

def register_input(controller, input, color):
    for i in input:
        controller[i] = color



if __name__ == '__main__':
    filename = get_path_of('saved_data/best/model_gen63.obj')
    filename = get_path_of('saved_data/result7/model_gen58.obj')
    config = load_config()

    with open(filename, 'rb') as file:
        genome = pickle.load(file)

    replay_genome(genome, right_movements)

    # make_controller(['right', 'A', 'B'], 5)

