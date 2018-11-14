import numpy as np
import math


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

    sensor_map = np.argmax(sensor_map, axis=2)

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
