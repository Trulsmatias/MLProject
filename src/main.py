import logging
import signal
from logger import setup_logging
import multiprocessing
from evolution import make_first_generation, make_child, create_next_generation, rank_selection
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
    plt.switch_backend("tkagg")  # must have for matplotlib to work on mac in this case

    # Set up logger
    setup_logging()
    log = logging.getLogger('MLProject')
    log.info('Starting MLProject')
    profiling.start()
    profiling.mem()

    simulation_params = SimulationParameters(
        movements=right_movements,
        state_space_shape=(10, 10),  # shape after cropping
        action_space_shape=len(right_movements),
        max_simulation_steps=10000,
        num_generations=1,
        num_individuals_per_gen=200,
        selection_func=rank_selection,
        num_parents_per_child=2,
        breeding_func=make_child,
        mutation_rate_individual=0.5,
        mutation_rate_genes=0.5,
        num_select=10,
        max_subseq_length=10,
        parallel=True,
        num_workers=3
    )
    simulation_params.load_from_file()

    # The Simulator object, which lets individuals play Mario.
    simulator = Simulator(simulation_params.movements, simulation_params.max_simulation_steps)
    if simulation_params.parallel:
        simulator = ParallelSimulator(simulator, simulation_params.num_workers)
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
        simulator.simulate_generation(current_generation, render=False)  # can set parameter render=True

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

    last_generation = current_generation
    individuals_sorted = sorted(last_generation.individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)
    data_collector.save_module(individuals_sorted[0])
    """
        for i in range(len(individuals_sorted)):
        individuals_sorted[i].agent.save_model('models/model_{}.h5'.format(i + 1))
        # util.save_to_file(individuals_sorted[i].agent.model, i + 1)

    """
