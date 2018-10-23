class Individual:
    def __init__(self, id, chromosomes):
        self.id = id
        self.fitness = 0
        self.chromosomes = chromosomes

    def __str__(self):
        return 'Individual(id={}, fitness={}, chromosomes={})'.format(self.id, self.fitness, self.chromosomes)

    def __repr__(self):
        return self.__str__()


class Generation:
    def __init__(self, num, individuals):
        self.num = num
        self.individuals = individuals
