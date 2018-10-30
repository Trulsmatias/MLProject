from evolution import make_first_generation, roulette_wheel_selection, make_child, _create_next_generation
import logging
import sys
from generations import EvolutionParameters
from movements import right_movements
from play import Simulator


if __name__ == '__main__':
    # Set up logger
    log_formatter = logging.Formatter('%(name)s: [%(levelname)s] %(message)s')  # %(module)s:%(funcName)s
    stdout_log_hander = logging.StreamHandler(sys.stdout)
    stdout_log_hander.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(stdout_log_hander)
    log = logging.getLogger('MLProject')
    log.info('Starting MLProject')

    # Constants controlling simulation and evolution
    STATE_SPACE_SHAPE = (20, 21, 3)
    ACTION_SPACE_SHAPE = len(right_movements)
    MAX_SIMULATION_STEPS = 500
    NUM_GENERATIONS = 100
    NUM_INDIVIDUALS_PER_GENERATION = 10  # For now, this should prob be increased
    evolution_params = EvolutionParameters(
        selection_func=roulette_wheel_selection,
        num_parents_per_child=2,
        breeding_func=make_child,
        mutation_rate=0.1,
        num_select=5
    )

    generations = []
    current_generation = make_first_generation(NUM_INDIVIDUALS_PER_GENERATION, STATE_SPACE_SHAPE, ACTION_SPACE_SHAPE)
    generations.append(current_generation)

    simulator = Simulator(right_movements, MAX_SIMULATION_STEPS)

    for i_generation in range(NUM_GENERATIONS):
        log.debug('Simulating generation {}'.format(current_generation.num))
        simulator.simulate_generation(current_generation)
        current_generation = _create_next_generation(current_generation, evolution_params)
        generations.append(current_generation)
