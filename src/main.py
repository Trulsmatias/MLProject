import logging
import signal
from logger import setup_logging
import multiprocessing
from evolution import make_first_generation, make_child_random_subsequence, \
    create_next_generation, rank_selection, roulette_wheel_selection, top_n_selection
from parallel.simulate import ParallelSimulator
from simple_data_collector import DataCollection
import profiling
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from generations import SimulationParameters
from movements import right_movements
from play import Simulator


def _anim_thread(simulator):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(_):
        if simulator.state_downscaled is not None:
            ax.imshow(simulator.state_downscaled)
            ax.grid(which='both', axis='both', linestyle='-', color='k', linewidth=1)
            ax.set_xticks(np.arange(-.5, simulation_params.state_space_shape[1], 1))
            ax.set_yticks(np.arange(-.5, simulation_params.state_space_shape[0], 1))
            ax.set_xticklabels(np.arange(0, simulation_params.state_space_shape[1] + 1, 1))
            ax.set_yticklabels(np.arange(simulation_params.state_space_shape[0], -1, -1))

    anim = animation.FuncAnimation(fig, animate, interval=500)
    plt.show()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Spawn fresh worker processes, don't use fork

    # Set up logger
    setup_logging()
    log = logging.getLogger('MLProject')
    log.info('Starting MLProject')
    profiling.start()
    profiling.mem()

    simulation_params = SimulationParameters(
        movements=right_movements,
        state_space_shape=(13, 10),  # shape after cropping
        action_space_shape=len(right_movements),
        max_simulation_steps=20000,
        num_generations=100,
        num_individuals_per_gen=500,
        selection_func=top_n_selection,
        num_parents_per_child=2,
        breeding_func=make_child_random_subsequence,
        mutation_rate_individual=0.5,
        mutation_rate_genes=0.3,
        num_select=125,
        max_subseq_length=10,
        parallel=True,
        num_workers=3,
        headless=False,
        render=False
    )

    simulation_params.load_from_file()
    log.info('Using these simulation parameters:')
    for key, value in simulation_params.get_all_params().items():
        log.info('- {}: {}'.format(key, value))

    if not simulation_params.headless:
        plt.switch_backend('tkagg')  # must have for matplotlib to work on mac in this case

    # The Simulator object, which lets individuals play Mario.
    simulator = Simulator(simulation_params.movements, simulation_params.max_simulation_steps)
    if simulation_params.parallel:
        simulator = ParallelSimulator(simulator, simulation_params)
    # threading.Thread(target=_anim_thread, args=(simulator,)).start()

    # Handler for shutting down workers when we press Stop
    def sigterm_handler(sig, sframe):
        log.info('Caught signal {}, shutting down'.format(sig))
        simulator.shutdown()

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    current_generation = make_first_generation(simulation_params)

    data_collector = DataCollection(simulation_params)

    for i_generation in range(simulation_params.num_generations):
        t_start = time.time()
        log.info('Simulating generation {}'.format(current_generation.num))
        simulator.simulate_generation(current_generation, simulation_params.render)  # can set parameter render=True

        data_collector.collect_data(current_generation)

        log.info('Breeding next generation')
        current_generation = create_next_generation(current_generation, simulation_params)

        log.info('Simulation of generation took {:.2f} seconds'.format(time.time() - t_start))
        # Show memory usage after each generation for finding memory leaks
        profiling.mem()

    # Show number of different objects in memory for finding memory leaks
    profiling.obj()

    # Shut down worker processes
    simulator.shutdown()
