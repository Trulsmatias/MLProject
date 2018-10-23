class Individual:
    def __init__(self, id, chromosomes):
        self.id = id
        self.fitness = 0
        self.chromosomes = chromosomes

    def __repr__(self):
        return 'Individual(id={}, fitness={}, chromosomes={})'.format(self.id, self.fitness, self.chromosomes)


class Generation:
    def __init__(self, num, individuals):
        self.num = num
        self.individuals = individuals

    def __repr__(self):
        individuals_str = ''
        if len(self.individuals) > 0:
            individuals_str = '\n    ' + '\n    '.join([str(i) for i in self.individuals]) + '\n'
        return 'Generaiton(num={}, children={})'.format(self.num, individuals_str)
