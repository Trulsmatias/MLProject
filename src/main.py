from evolution import make_first_generation, roulette_wheel_selection, make_child
from generations import EvolutionParameters
from movements import right_movements
from play import Simulator

if __name__ == '__main__':
    evolution_params = EvolutionParameters(
        selection_func=roulette_wheel_selection,
        num_parents_per_child=2,
        breeding_func=make_child,
        mutation_rate=0.1,
        num_select=5
    )
    NUM_INDIVIDUALS_PER_GENERATION = 10  # For now, this should prob be increased
    NUM_GENERATIONS = 100
    STATE_SPACE_SHAPE = (20, 21, 3)
    ACTION_SPACE_SHAPE = len(right_movements)
    MAX_SIMULATION_STEPS = 50000

    generations = []
    current_generation = make_first_generation(NUM_INDIVIDUALS_PER_GENERATION, STATE_SPACE_SHAPE, ACTION_SPACE_SHAPE)
    generations.append(current_generation)

    simulator = Simulator(right_movements, MAX_SIMULATION_STEPS)

    for i_generation in range(NUM_GENERATIONS):
        simulator.simulate_generation(current_generation)
        # TODO
