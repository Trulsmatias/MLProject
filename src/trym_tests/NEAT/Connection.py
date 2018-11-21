


class Connection:
    def __init__(self, in_node, out_node, type, weight, enabled, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight  # The weight of the Connection between Nodes
        self.enabled = enabled  # True or false
        self.innovation_number = innovation_number  # New unique number used to identify connection
        self.type = type  # 1 for in to hidden, 2 for hidden to out and 3 for in to out


    def set_disabled(self, metode):
        # print(metode, 'is disabeling:', self.in_node, self.out_node, self.innovation_number)
        self.enabled = False