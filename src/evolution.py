from generations import Generation, Individual
import numpy as np


def selection(generation):
    """
    Select n fittest individuals from the generation.
    :param generation:
    :return:
    """
    return None


def roulette_wheel_selection(generation: Generation, num_select: int):
    """
    Performs roulette wheel selection on individuals in the generation.
    :param generation:
    :param num_select: number of individuals to select
    :return: a list of individuals that were selected
    """
    individuals_sorted = sorted(generation.individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)
    fitness_sum = sum([individual.fitness for individual in individuals_sorted])
    probabilities = [individual.fitness / fitness_sum for individual in individuals_sorted]

    return np.random.choice(individuals_sorted, size=num_select, replace=False, p=probabilities)


def make_child(parents):
    return np.random.choice(parents)


def reproduce(individuals, num_parents_per_child, num_children_per_family):
    """

    :return:
    """

    return


def mutate():
    pass


if __name__ == '__main__':
    individuals = [
        Individual(1, 849),
        Individual(2, 849),
        Individual(3, 849)
    ]

    np.random.seed(42)
    for i in individuals:
        i.fitness = np.random.random()

    gen = Generation(1, individuals)
    print(gen.individuals)
    print(roulette_wheel_selection(gen, 1))
