import logging
from agent import NNAgent
from generations import Generation, Individual
from itertools import combinations as comb
import numpy as np
import math

_log = logging.getLogger('MLProject.evolution')


def roulette_wheel_selection(individuals, num_select):
    """
    Performs roulette wheel selection on the given individuals.
    :param individuals:
    :param num_select: number of individuals to select
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

    chosen = np.random.choice(individuals_sorted, size=num_select-1, replace=False, p=probabilities)
    chosen = chosen.tolist()
    chosen.append(best_chosen)
    _log.debug("The chosen ones:")
    for c in chosen:
        _log.debug(c)
    return chosen


def top_n_selection(individuals, num_select):
    """
    Select the top n individuals from the group
    :param individuals:
    :param num_select: number of individuals to select
    :return: a list of individuals that were selected
    """
    sorted_individuals = sorted(individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)

    sorted_individuals = sorted_individuals[0:num_select]
    _log.debug("The chosen ones:")
    for i in sorted_individuals:
        _log.debug(i)

    return sorted_individuals


def rank_selection(individuals, num_select):
    """
    Performs rank selection on the given individuals.
    :param individuals:
    :param num_select: number of individuals to select
    :return: a list of individuals that were selected
    """
    sorted_individuals = sorted(individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)

    best_chosen = sorted_individuals.pop(0)
    divider = sum(range(len(individuals)))
    probabilities = [(len(sorted_individuals) - i)/divider for i in range(len(sorted_individuals))]

    chosen = np.random.choice(sorted_individuals, size=num_select - 1, replace=False, p=probabilities)
    chosen = chosen.tolist()
    chosen.append(best_chosen)
    _log.debug("The chosen ones:")
    for c in chosen:
        _log.debug(c)
    return chosen


def make_child(parents):
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
        which_parent = (i_matrix % (skip * len(parents))) // skip
        weights[i_matrix] = parent_weights[which_parent][i_matrix]  # TODO: maybe optimize this
        weights[i_matrix + 1] = parent_weights[which_parent][i_matrix + 1]
    child.agent.set_weights(weights)

    return child


def make_child_magnus_test(parents):
    parent_chromosomes = [parent.chromosomes for parent in parents]
    avg = np.average(parent_chromosomes)
    variance = np.var(parent_chromosomes)
    chromosome = avg + (np.random.random() - 0.5) * variance

    print('avg: {}, var: {}, chromosome: {}'.format(avg, variance, chromosome))
    child = Individual(chromosome)
    child.fitness = np.random.random()
    return child


def _reproduce(parents, num_parents_per_family, total_children, breeding_func=make_child):
    """
    WARNING. SHOULD NOT BE USED!
    May currently produce the wrong number of children.

    Creates a number of children by reproduction.
    :param parents: the parents, aka. the fittest individuals after selection
    :param num_parents_per_family:
    :param total_children:
    :param breeding_func: the function to call to make a single child. Defaults to make_child(parents)
    :return: children of the next generation
    """
    children = []
    num_children_per_family = (total_children * num_parents_per_family) // len(parents)
    # TODO: may produce wrong number of children. Eks 5 parents and 10 children total
    for i in range(0, len(parents), num_parents_per_family):
        family_parents = parents[i:i + num_parents_per_family]
        for j in range(num_children_per_family):
            children.append(breeding_func(family_parents))

    if total_children != len(children):
        _log.debug("Feil antall barn produsert!")
        _log.debug("Skulle laget", total_children, "antall barn")
        _log.debug("Produserte: ", len(children))

    return children


def _reproduce_slice(parents, num_parents_per_family, total_children, breeding_func=make_child):
    """
    Creates a number of children by reproduction.
    Makes enough children, and then returns the number of children that is needed.
    They who are chosen are not nececerly the best children.
    :param parents: the parents, aka. the fittest individuals after selection
    :param num_parents_per_family:
    :param total_children:
    :param breeding_func: the function to call to make a single child. Defaults to make_child(parents)
    :return: children of the next generation
    """

    children = []

    if num_parents_per_family > len(parents):
        num_parents_per_family = len(parents)

    parents_sorted = sorted(parents, key=lambda parent: parent.fitness, reverse=True)  # Sort parents to get the best parents at the top

    families = list(comb(parents_sorted, num_parents_per_family))  # Every combination of families
    families = families[0:total_children]  # Slice to only get the top n (n = total_children) best parent combinations

    _log.debug("Parents must reproduce " + str(math.ceil(total_children / len(families))) + " batches of children")

    # Makes enough children
    while len(children) < total_children:
        for fam in list(families):
            child = breeding_func(fam)
            sum_fitness_parents = sum(p.fitness for p in fam)
            child.fitness = sum_fitness_parents
            children.append(child)

    # More children than expected
    if len(children) > total_children:
        children_sorted = sorted(children, key=lambda child: child.fitness, reverse=True)
        children = children_sorted[:total_children]

    return children


def _mutate(children, mutation_rate):
    """
    Mutates individuals (children) based on the mutation rate.
    :param children: a list of Individuals (the children) to mutate
    :param mutation_rate: the chance for a single individual to completely mutated
    """
    for child in children:
        if np.random.random() < mutation_rate:
            weights = child.agent.get_weights()
            for i_matrix in range(len(weights)):  # For each W and b matrix
                for i_weight, weight in np.ndenumerate(weights[i_matrix]):  # For each element in the matrix
                    if np.random.random() < mutation_rate:
                        weights[i_matrix][i_weight] = np.random.random() * 2 - 1
            child.agent.set_weights(weights)


def make_first_generation(num_individuals, state_space_shape, action_space_size):
    """
    Creates the first generation. Individuals have random weights.
    :param num_individuals:
    :param state_space_shape: for now: num pixels
    :param action_space_size:
    :return:
    """

    """
    from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
    """
    individuals = [Individual(NNAgent(state_space_shape, action_space_size)) for _ in range(num_individuals)]
    return Generation(1, individuals)


def create_next_generation(generation, evolution_parameters):
    """
    Creates next generation.
    Assumes that each individual has been assigned a fitness score.
    This includes:
      - selection of fittest individuals
      - reproduction (breed new individuals)
      - mutation of newly bred individuals
      - creation of a new generation
    :param generation: the generation to simulate and reproduce
    :param evolution_parameters:
    :return: the new generation
    """
    _log.debug('Selecting individuals to breed next gen')
    # TODO: fix this weird programming architecture? A callable object attribute which is not a method, what??
    selected = evolution_parameters.selection_func(generation.individuals,
                                                   evolution_parameters.num_select)

    _log.debug('Reproducing')
    children = _reproduce_slice(selected, evolution_parameters.num_parents_per_child,
                                len(generation.individuals) - len(selected), evolution_parameters.breeding_func)

    _log.debug('Mutating')
    _mutate(children, evolution_parameters.mutation_rate)

    new_individuals = selected + children
    return Generation(generation.num + 1, new_individuals)


if __name__ == '__main__':
    from movements import right_movements
    np_first_gen = 5
    ni_second_gen = 11
    np_fam = 2
    first_gen = make_first_generation(np_first_gen, (12, 13, 3), len(right_movements))
    c = _reproduce_slice(first_gen.individuals, np_fam, ni_second_gen)
    # print(c)
