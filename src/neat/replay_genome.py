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


def replay_genome(genome, movements, gen):
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

        #filename = get_path_of('all_pictures/mario/')
        #imsave(filename + 'mario_' + str(_) + '.png', state)

        save_state = np.full((13, 10, 3), 255, dtype=np.int)

        COLORS = [[250, 250, 250], [0, 0, 0], [196, 0, 0], [0, 0, 196]]

        for i in range(13):
            for j in range(10):
                if state_downscaled[(i, j)] == -1:
                    save_state[(i, j)] = COLORS[3]
                elif state_downscaled[(i, j)] == 0:
                    save_state[(i, j)] = COLORS[0]
                else:
                    save_state[(i, j)] = COLORS[1]

        save_state[(7, 2)] = COLORS[2]


        # filename = get_path_of('all_pictures/input_downscaled/')
        # imsave(filename + 'state_' + str(_) + '.png', save_state.astype(np.uint8))



        # make_controller(movements[action], _, gen)



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


def make_controller(game_inputs, step, gen):
    controller = np.full((10, 18, 3), 255, dtype=np.int)

    LIGHT_RED = [255, 223, 223]
    DARK_RED = [196, 0, 0]

    GREY = [220, 220, 220]
    BLACK = [0, 0, 0]

    UP = [(2, 3),(2, 4),(3, 3),(3, 4)]
    DOWN = [(6, 3),(6, 4),(7, 3),(7, 4)]
    LEFT = [(4, 1),(4, 2),(5, 1),(5, 2)]
    RIGHT = [(4, 5),(4, 6),(5, 5),(5, 6)]

    A = [(5, 15),(5, 16),(6, 15),(6, 16)]
    B = [(5, 11),(5, 12),(6, 11),(6, 12)]

    register_input(controller, UP, GREY)
    register_input(controller, DOWN, GREY)
    register_input(controller, LEFT, GREY)
    register_input(controller, RIGHT, GREY)
    register_input(controller, A, LIGHT_RED)
    register_input(controller, B, LIGHT_RED)

    for input in game_inputs:
        if input == 'right':
            register_input(controller, RIGHT, BLACK)
        elif input == 'A':
            register_input(controller, A, DARK_RED)
        elif input == 'B':
            register_input(controller, B, DARK_RED)

    # plt.imshow(controller)
    filename = get_path_of('all_pictures/' + gen + '/Controller/')
    # filename = get_path_of('all_pictures/test/')
    imsave(filename + 'controller_'+str(step)+'.png', controller.astype(np.uint8))

    # plt.show()

def register_input(controller, input, color):
    for i in input:
        controller[i] = color



if __name__ == '__main__':

    gen = 'gen1'

    # filename = get_path_of('make_videos/model_' + gen + '.obj')
    filename = get_path_of('saved_data/result20/model_' + gen + '.obj')
    config = load_config()

    with open(filename, 'rb') as file:
        genome = pickle.load(file)

    replay_genome(genome, right_movements, gen)

    # make_controller(['right', 'A', 'B'], 5)

