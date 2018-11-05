from evolution import make_first_generation, roulette_wheel_selection, make_child, create_next_generation
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import logging
import sys
from generations import EvolutionParameters
from movements import right_movements
from play import Simulator


if __name__ == '__main__':
    plt.switch_backend("tkagg")  # must have for matplotlib to work on mac in this case
    # Set up logger
    log_formatter = logging.Formatter('%(name)s: [%(levelname)s] %(message)s')  # %(module)s:%(funcName)s
    stdout_log_hander = logging.StreamHandler(sys.stdout)
    stdout_log_hander.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(stdout_log_hander)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    log = logging.getLogger('MLProject')
    log.info('Starting MLProject')

    # Constants controlling simulation and evolution
    STATE_SPACE_SHAPE = (12, 13, 3)  # shape after cutting
    ACTION_SPACE_SHAPE = len(right_movements)
    MAX_SIMULATION_STEPS = 10000  # For now. This should prob be increased
    NUM_GENERATIONS = 100
    NUM_INDIVIDUALS_PER_GENERATION = 20  # For now. This should prob be increased
    evolution_params = EvolutionParameters(
        selection_func=roulette_wheel_selection,
        num_parents_per_child=2,
        breeding_func=make_child,
        mutation_rate=0.05,
        num_select=4
    )

    generations = []
    current_generation = make_first_generation(NUM_INDIVIDUALS_PER_GENERATION, STATE_SPACE_SHAPE, ACTION_SPACE_SHAPE)
    generations.append(current_generation)

    simulator = Simulator(right_movements, MAX_SIMULATION_STEPS)

    def anim_thread(simulator):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        def animate(_):
            if simulator.state_downscaled is not None:
                ax.imshow(simulator.state_downscaled)
                ax.grid(which='both', axis='both', linestyle='-', color='k', linewidth=1)
                ax.set_xticks(np.arange(-.5, STATE_SPACE_SHAPE[1], 1))
                ax.set_yticks(np.arange(-.5, STATE_SPACE_SHAPE[0], 1))
                ax.set_xticklabels(np.arange(0, STATE_SPACE_SHAPE[1] + 1, 1))
                ax.set_yticklabels(np.arange(STATE_SPACE_SHAPE[0], -1, -1))

        anim = animation.FuncAnimation(fig, animate, interval=500)
        plt.show()


    threading.Thread(target=anim_thread, args=(simulator,)).start()

    for i_generation in range(NUM_GENERATIONS):
        log.debug('Simulating generation {}'.format(current_generation.num))
        simulator.simulate_generation(current_generation, render=True)  # can set parameter render=True
        log.debug('Breeding next generation')
        current_generation = create_next_generation(current_generation, evolution_params)
        generations.append(current_generation)
