from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import threading, time

import msvcrt
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env_mordi = gym_super_mario_bros.SuperMarioBrosEnv(frames_per_step=4, rom_mode='rectangle')
env = BinarySpaceToDiscreteSpaceEnv(env_mordi, SIMPLE_MOVEMENT)

input_space = {
    'A': 6,
    'D': 1,
    'S': 0,
    'W': 5,
}

done = True
key = b's'

def thread1():
    global key
    lock = threading.Lock()
    while True:
        with lock:
            key = msvcrt.getch()

threading.Thread(target = thread1).start()

for step in range(5000):
    time.sleep(0.1)
    a = key.decode("utf-8").upper()

    test = [env_mordi._read_mem(mem) for mem in [0x04B0, 0x04B1, 0x04B2, 0x04B3,
                                                 0x04B4, 0x04B5, 0x04B6, 0x04B7,
                                                 0x04B8, 0x04B9, 0x04BA, 0x04BB,
                                                 0x04BC, 0x04BD, 0x04BE, 0x04BF,
                                                 0x04C0, 0x04C1, 0x04C2, 0x04C3]]

    test = env_mordi._read_mem(0x071A)
    mario_pos = [env_mordi._read_mem(mem) for mem in [0x04AC, 0x04AD, 0x04AE, 0x04AF]]


    if done:
        state = env.reset()
    state, reward, done, info = env.step(input_space[a])
    env.render()


env.close()





