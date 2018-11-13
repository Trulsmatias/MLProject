from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import threading, time

import msvcrt
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env_mordi = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=1, rom_mode='rectangle')
env = BinarySpaceToDiscreteSpaceEnv(env_mordi, SIMPLE_MOVEMENT)

input_space = {
    'A': 6,
    'D': 1,
    'S': 0,
    'W': 5,
}

done = True
key = b's'

def downscale(env_expanded, state, grid_start=(-4, -2), grid_end=(1, 5), res=1):
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

    sensor_map = state[mario_center[1] + grid_start[0]*scaling*res:mario_center[1] + grid_end[0]*scaling*res + 1:scaling,
                       mario_center[0] + grid_start[1]*scaling*res:mario_center[0] + grid_end[1]*scaling*res + 1:scaling]

    sensor_map = np.argmax(sensor_map, axis=2)

    # if np.shape(sensor_map)[0] != (grid_end[0] - grid_start[0]) + 1: # Check if height of sensor map is correct


    # if np.shape(sensor_map)[1] != (grid_end[1] - grid_start[1]) + 1: # Check if width of sensor map is correct


    return sensor_map


def get_mario_pos(env_expanded):
    return [env_expanded._read_mem(mem) for mem in [0x04AC, 0x04AD, 0x04AE, 0x04AF]]

def thread1():
    global key
    lock = threading.Lock()
    while True:
        with lock:
            key = msvcrt.getch()

# threading.Thread(target = thread1).start()

time_start = time.time()

for step in range(500):
    # a = key.decode("utf-8").upper()
    #
    # test = [env_mordi._read_mem(mem) for mem in [0x04B0, 0x04B1, 0x04B2, 0x04B3,
    #                                              0x04B4, 0x04B5, 0x04B6, 0x04B7,
    #                                              0x04B8, 0x04B9, 0x04BA, 0x04BB,
    #                                              0x04BC, 0x04BD, 0x04BE, 0x04BF,
    #                                              0x04C0, 0x04C1, 0x04C2, 0x04C3]]
    #
    # test = env_mordi._read_mem(0x071A)
    # mario_pos = [env_mordi._read_mem(mem) for mem in [0x04AC, 0x04AD, 0x04AE, 0x04AF]]


    if done:
        state = env.reset()

    time1 = time.time()
    state, reward, done, info = env.step(env.action_space.sample())
    print("time", time.time() - time1)
    # env.render()

time_end = time.time()

fps = 500 / (time_end - time_start)
print("FPS:", fps)

env.close()





