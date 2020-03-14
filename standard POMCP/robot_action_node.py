import random as rand


class RobotActionNode:
    def __init__(self, env):
        """
        Initializes a robot action node.

        :param env: environment in which the robot and human operate
        :type env: Environment (for now)
        """
        self.type = "robot"
        self.env = env
        self.value = 0
        self.visited = 0
        self.human_node_children = self.init_children()

    def init_children(self):
        """
        Initializes all the human node children of this node to "empty".

        :return: initialized human node children
        :rtype: list
        """
        children = ["empty"] * len(self.env.observation_space)
        return children

    def augmented_value(self, c):
        """
        Returns the augmented value (value + exploration bonus) of taking this robot action.

        :return: augmented value of robot action
        :rtype: float
        """
        return self.value + float(c) / self.visited

    def update_value(self, reward):
        """
        Updates the value of the search node.

        :param reward: the immediate reward just received
        :type reward: float
        """
        self.value += (reward - self.value) / self.visited

    def update_visited(self):
        """
        Increments the number of times of visiting this node.
        """
        self.visited += 1
