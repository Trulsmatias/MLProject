import logging
from tensorflow import keras

from agent import NNAgent
from generations import Generation, Individual, SimulationParameters
from itertools import combinations as comb
import numpy as np
import math
from os import listdir
from os.path import isfile, join

_log = logging.getLogger('MLProject.evolution')


def roulette_wheel_selection(individuals, simulation_params):
    """
    Performs roulette wheel selection on the given individuals.
    :param individuals:
    :param simulation_params: all of the simulation params
    :return: a list of individuals that were selected
    """
    individuals_sorted = sorted(individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)
    best_chosen = individuals_sorted.pop(0)
    for individual in individuals:
        if individual.fitness <= 0:
            individual.fitness = 1  # cant have an 0 or negative probibility in roulettewheel. Dont wont the 1s anyway
    fitness_sum = sum([individual.fitness for individual in individuals_sorted])
    probabilities = [individual.fitness / fitness_sum for individual in individuals_sorted]

    chosen = np.random.choice(individuals_sorted, size=simulation_params.num_select-1, replace=False, p=probabilities)
    chosen = chosen.tolist()
    chosen.append(best_chosen)
    _log.debug("The chosen ones:")
    for c in chosen:
        _log.debug(c)
    return chosen


def top_n_selection(individuals, simulation_params):
    """
    Select the top n individuals from the group
    :param individuals:
    :param simulation_params: all of the simulation params
    :return: a list of individuals that were selected
    """
    sorted_individuals = sorted(individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)

    sorted_individuals = sorted_individuals[0:simulation_params.num_select]
    _log.debug("The chosen ones:")
    for i in sorted_individuals:
        _log.debug(i)

    return sorted_individuals


def rank_selection(individuals, simulation_params):
    """
    Performs rank selection on the given individuals.
    :param individuals:
    :param simulation_params: all of the simulation params
    :return: a list of individuals that were selected
    """
    sorted_individuals = sorted(individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)

    best_chosen = sorted_individuals[0]
    divider = sum(range(len(individuals) + 1))
    probabilities = [(len(sorted_individuals) - i)/divider for i in range(len(sorted_individuals))]

    num_select = simulation_params.num_select
    if num_select > len(sorted_individuals):
        num_select = len(sorted_individuals)
    chosen = np.random.choice(sorted_individuals, size=num_select - 1, replace=False, p=probabilities)
    chosen = chosen.tolist()
    chosen.append(best_chosen)
    _log.debug("The chosen ones:")
    for c in chosen:
        _log.debug(c)
    return chosen


def make_child(parents):
    # TODO make random, instead of alternating
    # TODO change every column istead of whole matrix
    """
    Make a single child from a list of parents.
    :param parents: a list of parent which will make a child
    :return: the child
    """
    child = Individual(NNAgent(parents[0].agent.state_space_shape, parents[0].agent.action_space_size))
    weights = child.agent.get_weights()

    parent_weights = [parent.agent.get_weights() for parent in parents]

    skip = 2  # this number will never change. Just for readability.
    for i_matrix in range(0, len(weights), skip):  # For each W and b matrix, alternating
        print("Num weights: ", len(weights))
        which_parent = (i_matrix % (skip * len(parents))) // skip
        weights[i_matrix] = parent_weights[which_parent][i_matrix]  # TODO: maybe optimize this
        weights[i_matrix + 1] = parent_weights[which_parent][i_matrix + 1]
    child.agent.set_weights(weights)

    return child


def _reproduce_slice(parents, simulation_params):
    """
    Creates a number of children by reproduction.
    Makes enough children, and then returns the number of children that is needed.
    They who are chosen are not nececerly the best children.
    :param parents: the parents, aka. the fittest individuals after selection
    :param simulation_params: all of the simulation params
    :return: children of the next generation
    """

    children = []

    num_parents_per_child = simulation_params.num_parents_per_child
    if num_parents_per_child > len(parents):
        num_parents_per_child = len(parents)

    parents_sorted = sorted(parents, key=lambda parent: parent.fitness, reverse=True)  # Sort parents to get the best parents at the top

    total_children = simulation_params.num_individuals_per_gen - simulation_params.num_select

    families = list(comb(parents_sorted, num_parents_per_child))  # Every combination of families
    families = families[0:total_children]  # Slice to only get the top n (n = total_children) best parent combinations

    batches_of_children = "0"
    if len(families) != 0:
        batches_of_children = str(math.ceil(total_children / len(families)))
    _log.debug("Parents must reproduce " + batches_of_children + " batches of children")

    # Makes enough children
    while len(children) < total_children:
        for fam in list(families):
            child = simulation_params.breeding_func(fam)
            sum_fitness_parents = sum(p.fitness for p in fam)
            child.fitness = sum_fitness_parents
            children.append(child)

    # More children than expected
    if len(children) > total_children:
        children_sorted = sorted(children, key=lambda child: child.fitness, reverse=True)
        children = children_sorted[:total_children]

    return children


def _mutate(children, simulation_params):
    # TODO fix
    """
    Mutates individuals (children) based on the mutation rate.
    :param children: a list of Individuals (the children) to mutate
    :param simulation_params: all of the simulation params
    """
    for child in children:
        if np.random.random() < simulation_params.mutation_rate_individual:
            weights = child.agent.get_weights()
            for i_matrix in range(len(weights)):  # For each W and b matrix
                for i_weight, weight in np.ndenumerate(weights[i_matrix]):  # For each element in the matrix
                    if np.random.random() < simulation_params.mutation_rate_genes:
                        weights[i_matrix][i_weight] = np.random.random() * 2 - 1

            child.agent.set_weights(weights)


def make_first_generation(simulation_params):
    """
    Creates the first generation. Individuals have random weights.
    :param simulation_params: all of the simulation params
    :return:
    """

    """
    from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
NUM_INDIVIDUALS_PER_GENERATION, STATE_SPACE_SHAPE, ACTION_SPACE_SHAPE
    """
    individuals = [Individual(NNAgent(
        simulation_params.state_space_shape,
        simulation_params.action_space_shape)) for _ in range(simulation_params.num_individuals_per_gen)]
    return Generation(1, individuals)


def create_next_generation(generation, simulation_params):
    """
    Creates next generation.
    Assumes that each individual has been assigned a fitness score.
    This includes:
      - selection of fittest individuals
      - reproduction (breed new individuals)
      - mutation of newly bred individuals
      - creation of a new generation
    :param generation: the generation to simulate and reproduce
    :param simulation_params: all of the simulation params
    :return: the new generation
    """
    _log.debug('Selecting individuals to breed next gen')
    # TODO: fix this weird programming architecture? A callable object attribute which is not a method, what??
    selected = simulation_params.selection_func(generation.individuals,
                                                simulation_params)

    _log.debug('Reproducing')
    children = _reproduce_slice(selected, simulation_params)

    _log.debug('Mutating')
    _mutate(children, simulation_params)

    new_individuals = selected + children
    return Generation(generation.num + 1, new_individuals)

def new_gen_with_challenger(filename, simulation_params):
    """
    Creates a new random generation with one individual from the outside.
    :param filename: where the file of the challenger is
    :param simulation_params: all of the simulation params
    :return: the new generation
    """
    state_space_shape = simulation_params.state_space_shape
    action_space_shape = simulation_params.action_space_shape
    gen = make_first_generation(simulation_params)
    challenger = NNAgent(state_space_shape, action_space_shape)
    challenger.load_model(filename)
    challenger = Individual(challenger)
    gen.add_individual(challenger)
    return gen


def continue_gen(path, simulation_params):
    """
    Creates a generation based on files from a directory.
    :param path: where the directory is
    :param simulation_params: all of the simulation params
    :return: the loaded generation
    """
    state_space_shape = simulation_params.state_space_shape
    action_space_shape = simulation_params.action_space_shape
    individuals = []
    filenames = [f for f in listdir(path) if isfile(join(path, f)) and not f == ".gitkeep"]
    for f in filenames:
        individual = NNAgent(state_space_shape, action_space_shape)
        individual.load_model(path + "/" + f)
        individuals.append(Individual(individual))
    return Generation(1, individuals)


if __name__ == '__main__':
    from movements import right_movements
    np_first_gen = 5
    ni_second_gen = 11
    np_fam = 2
    simul_params = SimulationParameters(
        state_space_shape=(10, 10),  # shape after cropping
        action_space_shape=len(right_movements),
        max_simulation_steps=10000,
        num_generations=1,
        num_individuals_per_gen=5,
        selection_func=rank_selection,
        num_parents_per_child=2,
        breeding_func=make_child,
        mutation_rate_individual=0.5,
        mutation_rate_genes=0.5,
        num_select=10
    )
    first_gen = make_first_generation(simul_params)
    c = _reproduce_slice(first_gen.individuals, np_fam, ni_second_gen)
    # print(c)
