from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import threading, time
import matplotlib.pyplot as plt
import matplotlib.colors as c
import skimage.measure
import math

import msvcrt
import numpy as np
import matplotlib.animation as animation

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env_mordi = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='vanilla')
env = BinarySpaceToDiscreteSpaceEnv(env_mordi, SIMPLE_MOVEMENT)

input_space = {
    b'a': 6,
    b'd': 1,
    b's': 0,
    b'w': 5,
}

key = b's'

def yas(array, axis):
    return array


def downscale(env_expanded, state, box_height=10, box_width=(-2, 7), res=1):
    """
    Returns a simplified grid around Mario.
    :param env_expanded: The environment that the gym runs in
    :param state: Super Mario game state
    :param grid_start: Where to start the sensor grid. Negative values indicating
                       how many 16x16 block above and behind to start the grid.
    :param grid_end: Where to end the sensor grid. Positive values indicating
                     how many 16x16 block underneath and in front to start the grid.
    :param res: The resolution of the grid. Standard is 1, making the blocks 16x16.
                Setting resolution to 2 will give 16x16 / 2 = 8x8.
    :return: sensor_map: 2D grid where each value represents what RGB value that
                         was highest in the selected sensor grid. 0 = red, 1 = green and 2 = blue.
    """
    mario_pos = get_mario_pos(env_expanded)

    scaling = int(16 / res)

    mario_center = [int((mario_pos[0] + mario_pos[2])/2), int((mario_pos[1] + mario_pos[3])/2)]

    if mario_center[0] == 0 and mario_center[1] == 0:
        mario_center[0] = 48
        mario_center[1] = 190



    sensor_map = state[6 + (15 - box_height)*scaling*res::scaling,
               mario_center[0] + box_width[0]*scaling*res:mario_center[0] + box_width[1]*scaling*res + 1:scaling]

    # sensor_map = state[80:255, 0:160]
    # sensor_map = np.dot(sensor_map[..., :3], [0.2126, 0.7152, 0.0722])

    # sensor_map = sensor_map.astype(int)

    sensor_map = np.argmax(sensor_map, axis=2)

    # if np.shape(sensor_map)[0] != (grid_end[0] - grid_start[0]) + 1: # Check if height of sensor map is correct

    # if np.shape(sensor_map)[1] != (grid_end[1] - grid_start[1]) + 1: # Check if width of sensor map is correct

    return sensor_map


def get_sensor_map(env_expanded, map_size=(13, 10), tiles_behind=2):
    state_map = np.zeros(map_size)
    from_range_x, to_range_x = (-tiles_behind) * 16, (map_size[1] - tiles_behind - 1) * 16
    from_range_y, to_range_y = -int((map_size[0] - 1) / 2) * 16, int((map_size[0] - 1) / 2) * 16
    mario_pos_in_map = (int((map_size[0] - 1) / 2), tiles_behind)  # (y, x)

    mario_x = (env_expanded._read_mem(0x6D) * 0x100 + env_expanded._read_mem(0x86)) + 8
    mario_y = env_expanded._read_mem(0x03B8)

    enemies_xy = []

    for slot in range(0, 5):
        enemy = env_expanded._read_mem(0xF + slot)

        if enemy != 0:
            ex = env_expanded._read_mem(0x6E + slot) * 0x100 + env_expanded._read_mem(0x87 + slot)
            ey = env_expanded._read_mem(0xCF + slot) + 24
            enemies_xy.append([ex, ey])

    row_counter = 0
    col_counter = 0
    for row in range(from_range_y, to_range_y + 1, 16):
        y_1 = mario_y + row
        for col in range(from_range_x, to_range_x + 1, 16):
            x_1 = mario_x + col
            suby = math.floor((y_1 - 32) / 16)
            subx = math.floor((x_1 % 256) / 16)
            page = math.floor(x_1 / 256) % 2
            addr = 0x500 + page * 13 * 16 + suby * 16 + subx

            if env_expanded._read_mem(addr) != 0 and suby < 13 and suby > 0:
                state_map[row_counter][col_counter] = 1

            col_counter += 1

        col_counter = 0
        row_counter += 1

    for enemy_pos in enemies_xy:
        enemy_distx = math.floor((enemy_pos[0] - mario_x) / 16)
        enemy_disty = math.floor(((enemy_pos[1] - 1) - mario_y) / 16)

        if -1 < enemy_distx + mario_pos_in_map[1] + 1 < map_size[1] and 0 <= enemy_disty + mario_pos_in_map[0] < map_size[0]:
            state_map[enemy_disty + mario_pos_in_map[0], enemy_distx + mario_pos_in_map[1] + 1] = -1

    # Insert Mario in state_map
    state_map[mario_pos_in_map[0] + 1, mario_pos_in_map[1]] = 3

    return state_map


def get_mario_pos(env_expanded):
    a = [env_expanded._read_mem(mem) for mem in [0x04AC, 0x04AD, 0x04AE, 0x04AF]]
    return [int((a[0] + a[2]) / 2), int((a[1] + a[3])/2)]

def thread1():
    global key
    lock = threading.Lock()
    while True:
        with lock:
            key = msvcrt.getch()

threading.Thread(target = thread1).start()


state = np.array([[-1, 1], [2, 3]])
done = True
x_pos = 0
last_x = 0
x_reward = 0
accumulated_fitness = 0

def update(i):
    global done
    global state
    global x_pos
    global x_reward
    global last_x
    global accumulated_fitness

    if done:
        state = env.reset()

    # state, reward, done, info = env.step(env.action_space.sample())
    state, reward, done, info = env.step(input_space[key])
    # state, reward, done, info = env.step(1)
    x_pos = info['x_pos'] + accumulated_fitness

    if info['flag_get']:
        accumulated_fitness += x_pos

    # matrice.set_array(get_sensor_map(env_mordi))

    env.render()



fig, ax = plt.subplots()
matrice = ax.imshow(state, vmax=state.max(), vmin=state.min())
plt.colorbar(matrice)



ani = animation.FuncAnimation(fig, update, frames=300, interval=1)
plt.show()


env.close()

