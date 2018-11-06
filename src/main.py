from evolution import make_first_generation, roulette_wheel_selection, make_child, create_next_generation, top_n_selection, rank_selection
import profiling

import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import logging
import sys
from generations import EvolutionParameters
from movements import right_movements
from play import Simulator
import util


def _anim_thread(simulator):
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


if __name__ == '__main__':
    plt.switch_backend("tkagg")  # must have for matplotlib to work on mac in this case

    # Set up logger
    log_formatter = logging.Formatter('%(asctime)s %(name)s: [%(levelname)s] %(message)s')  # %(module)s:%(funcName)s
    stdout_log_hander = logging.StreamHandler(sys.stdout)
    stdout_log_hander.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(stdout_log_hander)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    log = logging.getLogger('MLProject')
    log.info('Starting MLProject')
    profiling.start()
    profiling.mem()

    # Constants controlling simulation and evolution
    STATE_SPACE_SHAPE = (10, 10)  # shape after cropping
    ACTION_SPACE_SHAPE = len(right_movements)
    MAX_SIMULATION_STEPS = 10000  # For now. This should prob be increased
    NUM_GENERATIONS = 100
    NUM_INDIVIDUALS_PER_GENERATION = 50  # For now. This should prob be increased

    evolution_params = EvolutionParameters(
        selection_func=rank_selection,
        num_parents_per_child=2,
        breeding_func=make_child,
        mutation_rate=0.1,
        num_select=5
    )

    # The Simulator object, which lets individuals play Mario.
    simulator = Simulator(right_movements, MAX_SIMULATION_STEPS)
    # threading.Thread(target=_anim_thread, args=(simulator,)).start()

    current_generation = make_first_generation(NUM_INDIVIDUALS_PER_GENERATION, STATE_SPACE_SHAPE, ACTION_SPACE_SHAPE)

    for i_generation in range(NUM_GENERATIONS):
        log.debug('Simulating generation {}'.format(current_generation.num))
        simulator.simulate_generation(current_generation, render=False)  # can set parameter render=True

        log.debug('Breeding next generation')
        current_generation = create_next_generation(current_generation, evolution_params)

        # Show memory usage after each generation for finding memory leaks
        profiling.mem()

    # Show number of different objects in memory for finding memory leaks
    profiling.obj()

    last_generation = current_generation
    individuals_sorted = sorted(last_generation.individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)
    for i in range(len(individuals_sorted)):
        individuals_sorted[i].agent.save_model('models/model_{}.h5'.format(i + 1))
        # util.save_to_file(individuals_sorted[i].agent.model, i + 1)
