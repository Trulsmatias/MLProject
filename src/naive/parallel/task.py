class PickleSimulationTask:
    def __init__(self, id, weights, state_space_shape, action_space_size):
        self.id = id
        self.weights = weights
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size


class SimulationResult:
    def __init__(self, id, fitness):
        self.id = id
        self.fitness = fitness
