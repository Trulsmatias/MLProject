import logging
from agent import NNAgent
from generations import Generation, Individual
from itertools import combinations as comb
import numpy as np
import math

_log = logging.getLogger('MLProject')


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
    _log.info("The chosen ones:")
    for c in chosen:
        _log.info(c)
    return chosen


def make_child(parents):
    """
    Make a single child from a list of parents.
    :param parents: a list of parent which will make a child
    :return: the child
    """
    parent_agent = parents[0].agent
    child = Individual(NNAgent(parent_agent.state_space_shape, parent_agent.action_space_size))

    #child = copy.deepcopy(parents[0])  # Start with the first parent and add values from the other parents
    #child.id = 0  # "Reset" values inherited from the parent
    #child.fitness = 0
    #if len(parents) == 1:
    #    return child

    weights = child.agent.model.get_weights()
    skip = 2  # this number will never change. Just for readability.
    for i_matrix in range(len(weights), skip):  # For each W and b matrix, alternating
        which_parent = (i_matrix % (skip * len(parents))) / skip
        #if which_parent != 0:
        weights[i_matrix] = parents[which_parent].agent.model.get_weights()[i_matrix]  # TODO: maybe optimize this
        weights[i_matrix + 1] = parents[which_parent].agent.model.get_weights()[i_matrix + 1]
    child.agent.model.set_weights(weights)

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


def _reproduce2(parents, num_parents_per_family, total_children, breeding_func=make_child):
    """
    Creates a number of children by reproduction.
    :param parents: the parents, aka. the fittest individuals after selection
    :param num_parents_per_family:
    :param total_children:
    :param breeding_func: the function to call to make a single child. Defaults to make_child(parents)
    :return: children of the next generation
    """

    children = []
    families = list(comb(parents, 2))

    # print("Reproducing " + str(math.ceil(total_children / len(families))) + " times per parent batch")
    _log.info("Reproducing " + str(math.ceil(total_children / len(families))) + " times per parent batch")

    while len(children) < total_children:
        for i in list(families):
            children.append(breeding_func(i))

    # More children than expected
    if (len(children) > total_children):
        children = children[0:total_children]

    return children

def _mutate(children, mutation_rate):
    """
    Mutates individuals (children) based on the mutation rate.
    :param children: a list of Individuals (the children) to mutate
    :param mutation_rate: the chance for a single individual to completely mutated
    """
    for child in children:
        if np.random.random() < mutation_rate:
            weights = child.agent.model.get_weights()
            for i_matrix in range(len(weights)):  # For each W and b matrix
                for i_weight, weight in np.ndenumerate(weights[i_matrix]):  # For each element in the matrix
                    if np.random.random() < mutation_rate:
                        weights[i_matrix][i_weight] = np.random.random() * 2 - 1
            child.agent.model.set_weights(weights)


def make_first_generation(num_individuals, state_space_shape, action_space_size):
    """
    Creates the first generation. Individuals have random weights.
    :param num_individuals:
    :param state_space_shape: for now: num pixels
    :param action_space_size:
    :return:
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
    # TODO: fix this weird programming architecture? A callable object attribute which is not a method, what??
    selected = evolution_parameters.selection_func(generation.individuals,
                                                   evolution_parameters.num_select)
    children = _reproduce2(selected, evolution_parameters.num_parents_per_child,
                           len(generation.individuals) - len(selected), evolution_parameters.breeding_func)

    print(type(selected), type(children))

    _mutate(children, evolution_parameters.mutation_rate)

    new_individuals = selected + children
    return Generation(generation.num + 1, new_individuals)


if __name__ == '__main__':
    np.random.seed(45)

    individuals = []
    for i in range(12):
        chromosomes = np.random.random()
        individual = Individual(NNAgent((10, 10), 10))
        individual.fitness = np.random.random()
        individuals.append(individual)

    gen = Generation(1, individuals)

    for i in range(1):
        best = roulette_wheel_selection(gen.individuals, 4)
        children = _reproduce2(best, num_parents_per_family=2, total_children=23)
        #_mutate(children, 0.1)

        # print(gen)
        #gen = Generation(gen.num + 1, children)

        # print(gen.individuals)
        # print(roulette_wheel_selection(gen, 1))


