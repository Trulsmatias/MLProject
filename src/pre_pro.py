import numpy as np


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

    senesor_map = state[mario_center[1] + grid_start[0]*scaling*res:mario_center[1] + grid_end[0]*scaling*res + 1:scaling,
                        mario_center[0] + grid_start[1]*scaling*res:mario_center[0] + grid_end[1]*scaling*res + 1:scaling]

    senesor_map = np.argmax(senesor_map, axis=2)

    return senesor_map


def get_mario_pos(env_expanded):
    return [env_expanded._read_mem(mem) for mem in [0x04AC, 0x04AD, 0x04AE, 0x04AF]]


def distance_to_enemies(env_expanded):
    """
    Finds the distance up of up to five enemies
    :return [x-distance to enemy 1, y-distance to enemy 1, x-distance to enemy 2, y-distance to enemy 2, etc.:
    """
    enemy_pos = [env_expanded._read_mem(mem) for mem in [0x04B0, 0x04B1, 0x04B2, 0x04B3,
                                                         0x04B4, 0x04B5, 0x04B6, 0x04B7,
                                                         0x04B8, 0x04B9, 0x04BA, 0x04BB,
                                                         0x04BC, 0x04BD, 0x04BE, 0x04BF,
                                                         0x04C0, 0x04C1, 0x04C2, 0x04C3]]

    mario_pos = get_mario_pos(env_expanded)

    return [enemy_pos[2] - mario_pos[2], enemy_pos[1] - mario_pos[1],
            enemy_pos[6] - mario_pos[2], enemy_pos[5] - mario_pos[1],
            enemy_pos[10] - mario_pos[2], enemy_pos[9] - mario_pos[1],
            enemy_pos[14] - mario_pos[2], enemy_pos[13] - mario_pos[1],
            enemy_pos[18] - mario_pos[2], enemy_pos[17] - mario_pos[1]]
