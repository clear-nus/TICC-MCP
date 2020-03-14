import random as rand


class HumanActionNode:
    def __init__(self, env):
        """
        Initializes a human action node.

        :param env: environment in which the robot and human operate
        :type env: Environment (for now)
        """
        self.type = "human"
        self.env = env
        self.robot_node_children = self.init_children()
        self.value_list = self.init_value_list()
        self.visited_list = self.init_visited_list()
        self.belief = []

    def init_children(self):
        """
        Initializes all the robot node children of this node to "empty".

        :return: initialized robot node children
        :rtype: list
        """
        children = ["empty"] * len(self.env.action_space)
        return children

    def optimal_robot_action(self, c):
        """
        Returns the optimal robot action to take from this node.

        :param c: exploration constant
        :type c: float

        :return: optimal robot action
        :rtype: list
        """
        values = []
        for child in self.robot_node_children:
            if child == "empty":
                values.append(c)
            else:
                values.append(child.augmented_value(c))

        return self.env.action_space[values.index(max(values))]

    def init_value_list(self):
        """
        Creates the list which is used to store the value for each theta.

        :return: list of values for each theta
        :rtype: list
        """
        values = [0] * len(self.env.reward_space)
        return values

    def init_visited_list(self):
        """
        Creates the list storing number of times a particular theta value visited the node.

        :return: list of number of visits for each theta
        :rtype: list
        """
        visited_list = [0] * len(self.env.reward_space)
        return visited_list

    def update_value(self, reward, theta):
        """
        Updates the value of the search node.

        :param reward: the immediate reward just received
        :type reward: float
        :param theta: the theta visiting the node
        :type theta: integer
        """
        value = self.value_list[theta]
        self.value_list[theta] = value + \
            ((float(reward) - value) / self.visited_list[theta])

    def update_visited(self, theta):
        """
        Increments the number of times of visiting this node.

        :param theta: the theta visiting the node
        :type theta: integer
        """
        self.visited_list[theta] += 1

    def update_belief(self, augmented_state):
        """
        Add new augmented state perticle to the current belief set.

        :param augmented_state: the augmented state visiting this node
        :type augmented_state: list of world_state, theta and chi
        """
        self.belief.append(augmented_state)

    def sample_state(self):
        """
        Samples an augmented state from the current belief set.

        :return: a sampled augmented state
        :rtype: list of world_state, theta and chi
        """
        return rand.choice(self.belief)

    def get_children_values(self):
        """
        Returns the values of the robot children nodes of this node.

        :return: values of robot children nodes
        :rtype: list of float
        """
        values = [0] * len(self.robot_node_children)
        for i, child in enumerate(self.robot_node_children):
            if child != "empty":
                values[i] = child.value

        return values
